from dataclasses import dataclass
from typing import Callable, Tuple, Dict

import pandas as pd

from base_bot import BaseBot
from imc_types import OrderBook, Trade, Side, OrderRequest, Order
from prediction_model import PredictionModel, EisbachEnsemblePredictionModel, EisbachEnsembleFlowPredictionModel


__all__ = ['BavarianceBot']  # Export only the bot class

POSITION_LIMIT = 200


@dataclass
class ProductContext:
    product_name_1: str
    product_name_2: str
    prediction_model_1: PredictionModel
    prediction_model_2: PredictionModel
    formula_1: Callable[[float, float], float]
    formula_2: Callable[[float, float], float]
    """
    Small example usage:
    product_name_1 = '1_Eisbach'
    product_name_2 = '2_Eisbach_Call'
    prediction_model_1 = EisbachFlowRatePredictionModel()
    prediction_model_2 = EisbachWaterLevelPredictionModel()
    formula = lambda flow, level: (flow + level)
    """


class ProductContextManager:
    def __init__(self, ctx: ProductContext):
        self.ctx = ctx

    @staticmethod
    def _separate_orders(orderbook: OrderBook) -> Tuple[list, list, list, list]:
        """Separates own orders from market orders to avoid trading against oneself."""
        own_buy_orders = []
        others_buy_orders = []
        for buy_order in orderbook.buy_orders:
            if buy_order.volume == buy_order.own_volume:
                own_buy_orders.append(buy_order)
            else:
                others_buy_orders.append(Order(price=buy_order.price, volume=buy_order.volume - buy_order.own_volume, own_volume=0))

        own_sell_orders = []
        others_sell_orders = []
        for sell_order in orderbook.sell_orders:
            if sell_order.volume == sell_order.own_volume:
                # FIXED BUG: Was appending to own_buy_orders in original code
                own_sell_orders.append(sell_order)
            else:
                others_sell_orders.append(Order(price=sell_order.price, volume=sell_order.volume - sell_order.own_volume, own_volume=0))

        return own_buy_orders, others_buy_orders, own_sell_orders, others_sell_orders

    def _get_position(self, bot: BaseBot, product_name: str) -> int:
        positions = bot.request_positions() or {}
        return positions.get(product_name, 0)

    def handle_orderbook_update(self, orderbook: OrderBook, bot: BaseBot):
        """
        Entry point: Determines which product updated and calculates 
        the correct fair value using the shared models.
        """
        # 1. Get predictions from the underlying "Real Value" models
        # Note: Depending on your model logic, you might only want to step() 
        # the model once per tick, not per product update.
        settel_time = pd.Timestamp("2025-11-23 10:00")
        val_1 = self.ctx.prediction_model_1.predict_until_target(settel_time)
        val_2 = self.ctx.prediction_model_2.predict_until_target(settel_time)

        # 2. Determine which product we are trading and select the correct formula
       # if orderbook.product == "2_Eisbach_Call":
        #    return

        if orderbook.product == self.ctx.product_name_1:
            print("bruh")
            fair_value = self.ctx.formula_1(val_1, val_2)
            self._process_single_product(orderbook, bot, fair_value)


            # Marktwert < fair_value | > fair_value
            
        #elif orderbook.product == self.ctx.product_name_2:
         #   fair_value = self.ctx.formula_2(val_1, val_2)
          #  self._process_single_product(orderbook, bot, fair_value)

    def _process_single_product(self, orderbook: OrderBook, bot: BaseBot, predicted_value: float):
        """
        Generic logic to cancel orders, check limits, and place new orders 
        based on a calculated predicted_value.
        """
        print(f"Predicted value for {orderbook.product} = {predicted_value}")
        product = orderbook.product
        own_buy, others_buy, own_sell, others_sell = self._separate_orders(orderbook)
        
        current_position = self._get_position(bot, product)
        tick = orderbook.tick_size
        
        best_bid = others_buy[0].price if others_buy else None
        best_ask = others_sell[0].price if others_sell else None

        THRESHOLD = 2 * tick
        ORDER_SIZE = 10 

        # --- 1. Cancel existing orders ---
        for o in own_buy:
            bot.cancel_order(product, o.price)
        for o in own_sell:
            bot.cancel_order(product, o.price)

        # --- 2. Helper Checks ---
        def can_buy(size: int) -> bool:
            return current_position + size <= POSITION_LIMIT

        def can_sell(size: int) -> bool:
            return current_position - size >= -POSITION_LIMIT

        # --- 3. Trading Logic ---
        
        # Case A: Empty Book (Quote around prediction)
        if best_bid is None and best_ask is None:
            bid_price = predicted_value - tick
            ask_price = predicted_value + tick
            
            if can_buy(ORDER_SIZE):
                bot.send_order(OrderRequest(product, round(bid_price), Side.BUY, ORDER_SIZE))
            if can_sell(ORDER_SIZE):
                bot.send_order(OrderRequest(product, round(ask_price), Side.SELL, ORDER_SIZE))
            return

        # Case B: Only Asks exist (Place bid below)
        if best_bid is None and best_ask is not None:
            bid_price = best_ask - 2 * tick
            if can_buy(ORDER_SIZE):
                bot.send_order(OrderRequest(product, round(bid_price), Side.BUY, ORDER_SIZE))
            return

        # Case C: Only Bids exist (Place ask above)
        if best_ask is None and best_bid is not None:
            ask_price = best_bid + 2 * tick
            if can_sell(ORDER_SIZE):
                bot.send_order(OrderRequest(product, round(ask_price), Side.SELL, ORDER_SIZE))
            return

        # Case D: Market Active - Directional Trading
        mid = (best_bid + best_ask) / 2
        diff = predicted_value - mid

        # Bullish signal
        if diff > THRESHOLD and can_buy(ORDER_SIZE):
            bid_price = mid - tick # Try to front-run or join best bid
            bot.send_order(OrderRequest(product, round(bid_price), Side.BUY, ORDER_SIZE))

        # Bearish signal
        elif diff < -THRESHOLD and can_sell(ORDER_SIZE):
            ask_price = mid + tick
            bot.send_order(OrderRequest(product, round(ask_price), Side.SELL, ORDER_SIZE))

        # Else: Inside threshold, hold position.


class BavarianceBot(BaseBot):
    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)

        self.product_managers: Dict[str, ProductContextManager] = {}
        self._initialize_managers()

    def _initialize_managers(self):
        """
        Setup the 4 pairs covering the 8 products.
        """
        # ======================================================================
        # PAIR 1: Eisbach & Eisbach Call
        # ======================================================================
        # TODO: Ensure these paths are correct
        model_eisbach_level= EisbachEnsemblePredictionModel("../data/waterlevel_beginn_bis_31.12.2024_ezw_0.csv")
        model_eisbach_flow = EisbachEnsembleFlowPredictionModel("../data/flowrate_beginn_bis_31.12.2024_ezw_0.csv") # Example

        ctx_eisbach = ProductContext(
            product_name_1="1_Eisbach",
            product_name_2="2_Eisbach_Call",
            prediction_model_1=model_eisbach_flow,
            prediction_model_2=model_eisbach_level,
            # TODO: Update formulas
            formula_1=lambda flow, level: round(flow*level),  # 1_Eisbach
            formula_2=lambda flow, level: (flow + level) * 1.1  # 2_Eisbach_Call
        )
        mgr_eisbach = ProductContextManager(ctx_eisbach)
        self.product_managers["1_Eisbach"] = mgr_eisbach
        self.product_managers["2_Eisbach_Call"] = mgr_eisbach

        # ======================================================================
        # PAIR 2: Weather (3 & 4)
        # ======================================================================
        # TODO: Initialize actual models
        # ctx_weather = ProductContext(
        #     product_name_1="3_Weather",
        #     product_name_2="4_Weather",
        #     prediction_model_1=model_eisbach_flow, # Placeholder
        #     prediction_model_2=model_eisbach_flow, # Placeholder
        #     formula_1=lambda x, y: x * y, 
        #     formula_2=lambda x, y: x * y + 10
        # )
        # mgr_weather = ProductContextManager(ctx_weather)
        # self.product_managers["3_Weather"] = mgr_weather
        # self.product_managers["4_Weather"] = mgr_weather

        # # ======================================================================
        # # PAIR 3: Flights & Airport (5 & 6)
        # # ======================================================================
        # ctx_travel = ProductContext(
        #     product_name_1="5_Flights",
        #     product_name_2="6_Airport",
        #     prediction_model_1=model_eisbach_flow, # Placeholder
        #     prediction_model_2=model_eisbach_flow, # Placeholder
        #     formula_1=lambda x, y: x + 50,
        #     formula_2=lambda x, y: y + 50
        # )
        # mgr_travel = ProductContextManager(ctx_travel)
        # self.product_managers["5_Flights"] = mgr_travel
        # self.product_managers["6_Airport"] = mgr_travel

        # # ======================================================================
        # # PAIR 4: ETF & Strangle (7 & 8)
        # # ======================================================================
        # ctx_etf = ProductContext(
        #     product_name_1="7_ETF",
        #     product_name_2="8_ETF_Strangle",
        #     prediction_model_1=model_eisbach_flow, # Placeholder
        #     prediction_model_2=model_eisbach_flow, # Placeholder
        #     formula_1=lambda x, y: (x + y) / 2,
        #     formula_2=lambda x, y: abs(x - y)
        # )
        # mgr_etf = ProductContextManager(ctx_etf)
        # self.product_managers["7_ETF"] = mgr_etf
        # self.product_managers["8_ETF_Strangle"] = mgr_etf

    def on_trades(self, trades: list[dict]):
        """
        Handle trade updates if necessary. 
        Currently mostly for logging or tracking PnL.
        """
        pass

    def on_orderbook(self, orderbook: OrderBook):
        """
        Route orderbook updates to the correct ProductContextManager.
        """
        manager = self.product_managers.get(orderbook.product)
        if manager is not None:
            manager.handle_orderbook_update(orderbook, self)
        else:
            print(f"Warning: Received orderbook for unknown product {orderbook.product}")
