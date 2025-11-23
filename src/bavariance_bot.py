from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List
import time
import logging  # <--- ADDED

from base_bot import BaseBot
from imc_types import OrderBook, Trade, Side, OrderRequest, Order
from prediction_model import PredictionModel, EisbachEnsemblePredictionModel, EisbachEnsembleFlowPredictionModel

__all__ = ['BavarianceBot']

# Setup a logger for this specific file
logger = logging.getLogger(__name__)

POSITION_LIMIT = 300

# Extra ticks to add on all ASK (sell) quotes to make sells more aggressive (higher prices)
ASK_PREMIUM_TICKS = 1  # main sell edge in ticks

# Inventory-hedging parameters
BUY_EDGE_TICKS = 3                 # distance from mid / fair for hedge buys
SELL_EDGE_TICKS = ASK_PREMIUM_TICKS  # distance from mid / fair for main sells

HEDGE_START_SHORT = -int(0.5 * POSITION_LIMIT)  # start hedging once short beyond half the limit
HEDGE_ORDER_SIZE = 2                # small buy size for hedging (compared to ORDER_SIZE=10)

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
        self.orderbooks: Dict[str, OrderBook] = {}

    def update_orderbook(self, orderbook: OrderBook):
        """Updates the internal state with the latest market data."""
        self.orderbooks[orderbook.product] = orderbook

    @staticmethod
    def _separate_orders(orderbook: OrderBook) -> Tuple[List, List, List, List]:
        """Separates own orders from market orders."""
        own_buy, others_buy = [], []
        for order in orderbook.buy_orders:
            if order.volume == order.own_volume:
                own_buy.append(order)
            else:
                # Create virtual order with remaining volume
                others_buy.append(Order(price=order.price, volume=order.volume - order.own_volume, own_volume=0))

        own_sell, others_sell = [], []
        for order in orderbook.sell_orders:
            if order.volume == order.own_volume:
                own_sell.append(order)
            else:
                others_sell.append(Order(price=order.price, volume=order.volume - order.own_volume, own_volume=0))

        return own_buy, others_buy, own_sell, others_sell

    def execute_trading_logic(self, bot: BaseBot):
        """
        The active trading entry point. 
        Calculates predictions and executes orders based on the latest known state.
        """
        # Debug level for granular loop info
        logger.debug(f"[LOGIC] Executing trading logic for context: {self.ctx.product_name_1} & {self.ctx.product_name_2}")
        
        # 1. Get predictions (assuming models handle their own time/state steps)
        val_1 = self.ctx.prediction_model_1.predict_next_time_step()
        val_2 = self.ctx.prediction_model_2.predict_next_time_step()
        
        logger.debug(f"[LOGIC] Raw Predictions: Val1={val_1}, Val2={val_2}")

        # 2. Execute logic for Product 1
        if self.ctx.product_name_1 in self.orderbooks:
            ob_1 = self.orderbooks[self.ctx.product_name_1]
            predicted_fair_value_1 = self.ctx.formula_1(val_1, val_2)
            logger.info(f"[LOGIC] {self.ctx.product_name_1}: Calculated Fair Value = {predicted_fair_value_1}")
            self._process_single_product(ob_1, bot, predicted_fair_value_1)
        else:
            logger.warning(f"[LOGIC] {self.ctx.product_name_1}: No Orderbook found. Skipping.")

        # 3. Execute logic for Product 2 (currently disabled)
        # if self.ctx.product_name_2 in self.orderbooks:
        #     ob_2 = self.orderbooks[self.ctx.product_name_2]
        #     predicted_fair_value_2 = self.ctx.formula_2(val_1, val_2)
        #     self._process_single_product(ob_2, bot, predicted_fair_value_2)

    def _process_single_product(self, orderbook: OrderBook, bot: BaseBot, predicted_value: float):
        """
        Generic logic to cancel orders, check limits, and place new orders 
        based on a calculated predicted_value.
        """
        logger.debug(f"[STRATEGY] Processing {orderbook.product} | FairValue: {predicted_value:.2f}")
        product = orderbook.product
        own_buy, others_buy, own_sell, others_sell = self._separate_orders(orderbook)
        
        current_position = bot.positions.get(product, 0)
        logger.info(f"[STRATEGY] {product} Current Position: {current_position} (Limit: {POSITION_LIMIT})")
        
        tick = orderbook.tick_size
        
        best_bid = others_buy[0].price if others_buy else None
        best_ask = others_sell[0].price if others_sell else None
        logger.debug(f"[STRATEGY] Market State: Best Bid={best_bid}, Best Ask={best_ask}")

        THRESHOLD = 2 * tick
        ORDER_SIZE = 10  # main trade size

        # --- 1. Cancel existing orders ---
        if own_buy or own_sell:
            logger.info(f"[STRATEGY] Cancelling {len(own_buy)} buy orders and {len(own_sell)} sell orders")
            for o in own_buy:
                bot.cancel_order(product, o.price)
            for o in own_sell:
                bot.cancel_order(product, o.price)

        # --- 2. Helper Checks ---
        def can_buy(size: int) -> bool:
            allowed = current_position + size <= POSITION_LIMIT
            if not allowed:
                logger.warning(f"[STRATEGY] Cannot BUY {size}. Resulting Pos {current_position + size} > {POSITION_LIMIT}")
            return allowed

        def can_sell(size: int) -> bool:
            allowed = current_position - size >= -POSITION_LIMIT
            if not allowed:
                logger.warning(f"[STRATEGY] Cannot SELL {size}. Resulting Pos {current_position - size} < {-POSITION_LIMIT}")
            return allowed

        # --- 3. Trading Logic ---

        # Case A: Empty Book (Quote around prediction)
        if best_bid is None and best_ask is None:
            logger.info(f"[STRATEGY] DECISION: Case A (Empty Book). Quoting spread around {predicted_value}")
            # Use asymmetric edges: buy closer, sell further
            bid_price = predicted_value - BUY_EDGE_TICKS * tick
            ask_price = predicted_value + SELL_EDGE_TICKS * tick
            
            if can_buy(ORDER_SIZE):
                logger.info(f"[STRATEGY] -> Sending BUY order @ {round(bid_price)}")
                bot.send_order(OrderRequest(product, round(bid_price), Side.BUY, ORDER_SIZE))
            if can_sell(ORDER_SIZE):
                logger.info(f"[STRATEGY] -> Sending SELL order @ {round(ask_price)}")
                bot.send_order(OrderRequest(product, round(ask_price), Side.SELL, ORDER_SIZE))
            return

        # Case B: Only Asks exist (Place bid below)
        if best_bid is None and best_ask is not None:
            # Still just buying; helps reduce shorts naturally
            bid_price = best_ask - 2 * tick
            logger.info(f"[STRATEGY] DECISION: Case B (Only Asks). Placing Bid below best ask ({best_ask}) -> {bid_price}")
            if can_buy(ORDER_SIZE):
                logger.info(f"[STRATEGY] -> Sending BUY order @ {round(bid_price)}")
                bot.send_order(OrderRequest(product, round(bid_price), Side.BUY, ORDER_SIZE))
            return

        # Case C: Only Bids exist (Place ask above)
        if best_ask is None and best_bid is not None:
            # Main sell quote with edge SELL_EDGE_TICKS above best bid
            ask_price = best_bid + SELL_EDGE_TICKS * tick
            logger.info(
                f"[STRATEGY] DECISION: Case C (Only Bids). "
                f"Placing Ask above best bid ({best_bid}) with premium -> {ask_price}"
            )
            if can_sell(ORDER_SIZE):
                logger.info(f"[STRATEGY] -> Sending SELL order @ {round(ask_price)}")
                bot.send_order(OrderRequest(product, round(ask_price), Side.SELL, ORDER_SIZE))
            # Optional: inventory hedge if very short
            if current_position <= HEDGE_START_SHORT and can_buy(HEDGE_ORDER_SIZE):
                hedge_bid = best_bid - BUY_EDGE_TICKS * tick
                logger.info(
                    f"[STRATEGY] Inventory hedge (short={current_position}). "
                    f"Placing small BUY @ {round(hedge_bid)}"
                )
                bot.send_order(OrderRequest(product, round(hedge_bid), Side.BUY, HEDGE_ORDER_SIZE))
            return

        # Case D: Market Active - Directional Trading
        mid = (best_bid + best_ask) / 2
        diff = predicted_value - mid
        logger.debug(f"[STRATEGY] Analysis: Mid={mid}, Diff (PV - Mid)={diff:.2f}, Threshold={THRESHOLD}")

        placed_directional_buy = False
        placed_directional_sell = False

        # Bullish signal: main buy
        if diff > THRESHOLD and can_buy(ORDER_SIZE):
            logger.info(f"[STRATEGY] DECISION: Bullish (Diff > Threshold). Buying.")
            bid_price = mid - BUY_EDGE_TICKS * tick  # buy close to mid for good fill prob
            logger.info(f"[STRATEGY] -> Sending BUY order @ {round(bid_price)}")
            bot.send_order(OrderRequest(product, round(bid_price), Side.BUY, ORDER_SIZE))
            placed_directional_buy = True

        # Bearish signal: main sell
        elif diff < -THRESHOLD and can_sell(ORDER_SIZE):
            logger.info(f"[STRATEGY] DECISION: Bearish (Diff < -Threshold). Selling.")
            # Sell further away from mid for better edge
            ask_price = mid + SELL_EDGE_TICKS * tick
            logger.info(f"[STRATEGY] -> Sending SELL order @ {round(ask_price)}")
            bot.send_order(OrderRequest(product, round(ask_price), Side.SELL, ORDER_SIZE))
            placed_directional_sell = True

        else:
            logger.debug(f"[STRATEGY] DECISION: Signal within threshold; using inventory hedge only if needed.")

        # --- 4. Inventory hedge: small buys when too short ---
        if current_position <= HEDGE_START_SHORT and can_buy(HEDGE_ORDER_SIZE):
            # Only add extra hedge if we didn't already do a big directional buy this cycle
            if not placed_directional_buy:
                hedge_bid = mid - BUY_EDGE_TICKS * tick
                logger.info(
                    f"[STRATEGY] Inventory hedge: position {current_position} <= {HEDGE_START_SHORT}. "
                    f"Placing small BUY @ {round(hedge_bid)}"
                )
                bot.send_order(OrderRequest(product, round(hedge_bid), Side.BUY, HEDGE_ORDER_SIZE))

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
        model_eisbach_level = EisbachEnsemblePredictionModel("../data/waterlevel_beginn_bis_31.12.2024_ezw_0.csv")
        model_eisbach_flow = EisbachEnsembleFlowPredictionModel("../data/flowrate_beginn_bis_31.12.2024_ezw_0.csv")

        ctx_eisbach = ProductContext(
            product_name_1="1_Eisbach",
            product_name_2="2_Eisbach_Call",
            prediction_model_1=model_eisbach_flow,
            prediction_model_2=model_eisbach_level,
            # TODO: Update formulas
            formula_1=lambda flow, level: round(flow * level),  # 1_Eisbach
            formula_2=lambda flow, level: (flow + level) * 1.1  # 2_Eisbach_Call, TODO: Add correct formula
        )
        mgr_eisbach = ProductContextManager(ctx_eisbach)
        self.product_managers["1_Eisbach"] = mgr_eisbach
        self.product_managers["2_Eisbach_Call"] = mgr_eisbach

        # ======================================================================
        # PAIR 2: Weather (3 & 4) - placeholders commented out
        # ======================================================================
        # ctx_weather = ProductContext(
        #     product_name_1="3_Weather",
        #     product_name_2="4_Weather",
        #     prediction_model_1=model_eisbach_flow,  # Placeholder
        #     prediction_model_2=model_eisbach_flow,  # Placeholder
        #     formula_1=lambda x, y: x * y, 
        #     formula_2=lambda x, y: x * y + 10
        # )
        # mgr_weather = ProductContextManager(ctx_weather)
        # self.product_managers["3_Weather"] = mgr_weather
        # self.product_managers["4_Weather"] = mgr_weather

        # ======================================================================
        # PAIR 3: Flights & Airport (5 & 6) - placeholders commented out
        # ======================================================================
        # ctx_travel = ProductContext(
        #     product_name_1="5_Flights",
        #     product_name_2="6_Airport",
        #     prediction_model_1=model_eisbach_flow,  # Placeholder
        #     prediction_model_2=model_eisbach_flow,  # Placeholder
        #     formula_1=lambda x, y: x + 50,
        #     formula_2=lambda x, y: y + 50
        # )
        # mgr_travel = ProductContextManager(ctx_travel)
        # self.product_managers["5_Flights"] = mgr_travel
        # self.product_managers["6_Airport"] = mgr_travel

        # ======================================================================
        # PAIR 4: ETF & Strangle (7 & 8) - placeholders commented out
        # ======================================================================
        # ctx_etf = ProductContext(
        #     product_name_1="7_ETF",
        #     product_name_2="8_ETF_Strangle",
        #     prediction_model_1=model_eisbach_flow,  # Placeholder
        #     prediction_model_2=model_eisbach_flow,  # Placeholder
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
        for trade in trades:
            if trade.buyer == self.username or trade.seller == self.username:
                # Trade execution is critical info, so we use INFO or WARNING
                logger.info(f" [EXECUTION] {trade.product}: {trade.side} {trade.volume} @ {trade.price}")

    def on_orderbook(self, orderbook: OrderBook):
        """
        Route orderbook updates to the correct ProductContextManager.
        """
        manager = self.product_managers.get(orderbook.product)
        if manager:
            manager.update_orderbook(orderbook)
        else:
            # Warning level for data integrity issues
            logger.warning(f"Warning: Unknown product {orderbook.product}")

    def run_trading_loop(self):
        # Set interval to 10 secs
        INTERVAL = 10 
        
        while True:
            start_time = time.time()
            logger.info(f"\n--- Starting Trading Cycle: {time.ctime(start_time)} ---")

            # 1. UPDATE POSITIONS (Source of Truth)
            server_positions = self.request_positions()
            if server_positions is not None:
                self.positions = server_positions
                logger.info(f"Current Positions: {self.positions}")
            else:
                logger.error("CRITICAL: Failed to fetch positions. Skipping cycle to prevent errors.")
                continue

            # 2. EXECUTE STRATEGY
            unique_managers = set(self.product_managers.values())
            for manager in unique_managers:
                try:
                    manager.execute_trading_logic(self)
                except Exception as e:
                    logger.exception("Error in strategy execution")  # Will print full traceback automatically

            # 3. SLEEP UNTIL NEXT CYCLE
            elapsed = time.time() - start_time
            time.sleep(max(0, INTERVAL - elapsed))
