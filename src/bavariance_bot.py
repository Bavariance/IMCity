from base_bot import BaseBot
from imc_types import OrderBook, Trade, Side, OrderRequest

from prediction_model import PredictionModel, EisbachEnsemblePredictionModel


__all__ = ['BavarianceBot']  # Export only the bot class


PRODUCT_MODELS = {
    "1_Eisbach": EisbachEnsemblePredictionModel("../data/waterlevel_beginn_bis_31.12.2024_ezw_0.csv"),
    # "2_Eisbach_Call": None,
    # "3_Weather": None,
    # "4_Weather": None,
    # "5_Flights": None,
    # "6_Airport": None,
    # "7_ETF": None,
    # "8_ETF_Strangle": None,
}


POSITION_LIMIT = 200
class ProductManager:

    def __init__(self, product: str, prediction_model: PredictionModel):
        self.product = product
        self.prediction_model = prediction_model
        self.realtime_flag: bool = False

    @staticmethod
    def _separate_orders(orderbook):
        # Filter orders based on owner
        own_buy_orders = []
        others_buy_orders = []
        for buy_order in orderbook.buy_orders:
            if buy_order.volume == buy_order.own_volume:
                own_buy_orders.append(buy_order)
            else:
                # Subtract own volume
                buy_order.volume -= buy_order.own_volume
                others_buy_orders.append(buy_order)

        own_sell_orders = []
        others_sell_orders = []
        for sell_order in orderbook.sell_orders:
            if sell_order.volume == sell_order.own_volume:
                own_buy_orders.append(sell_order)
            else:
                # Subtract own volume
                sell_order.volume -= sell_order.own_volume
                others_buy_orders.append(sell_order)

        return own_buy_orders, others_buy_orders, own_sell_orders, others_sell_orders

    def _get_position(self, bot: BaseBot) -> int:
        positions = bot.request_positions() or {}
        return positions.get(self.product, 0)

    def handle_orderbook(self, orderbook: OrderBook, bot: BaseBot):
        own_buy_orders, others_buy_orders, own_sell_orders, others_sell_orders = self._separate_orders(orderbook)

        # TODO: Add timer?
        # TODO: Check if we have realtime data

        if realtime_flag:
            # TODO: Implement new realtime data logic

            realtime_flag = False
        else:
            # Get the best buy and sell offers
            best_bid = others_buy_orders[0].price
            best_ask = others_sell_orders[0].price

            predicted_value = self.prediction_model.predict_next_time_step()
            # TODO: Implement old data prediction logic
            current_position = self._get_position(bot)
            tick = orderbook.tick_size

            best_bid = orderbook.buy_orders[0].price if orderbook.buy_orders else None
            best_ask = orderbook.sell_orders[0].price if orderbook.sell_orders else None

            THRESHOLD = 2 * tick
            ORDER_SIZE = 10  # must divide your limit nicely (e.g. 20 * 10 = 200)

            # cancel existing own orders for this product
            for o in own_buy_orders:
                bot.cancel_order(self.product, o.price)
            for o in own_sell_orders:
                bot.cancel_order(self.product, o.price)

            predicted_value = self.prediction_model.predict_next_time_step()

            # helper to respect position limits
            def can_buy(size: int) -> bool:
                return current_position + size <= POSITION_LIMIT

            def can_sell(size: int) -> bool:
                return current_position - size >= -POSITION_LIMIT

            # empty book: quote around prediction, respecting limits
            if best_bid is None and best_ask is None:
                bid_price = predicted_value - tick
                ask_price = predicted_value + tick

                if can_buy(ORDER_SIZE):
                    buy_order = OrderRequest(
                        product=self.product,
                        price=bid_price,
                        side=Side.BUY,
                        volume=ORDER_SIZE,
                    )
                    bot.send_order(buy_order)

                if can_sell(ORDER_SIZE):
                    sell_order = OrderRequest(
                        product=self.product,
                        price=ask_price,
                        side=Side.SELL,
                        volume=ORDER_SIZE,
                    )
                    bot.send_order(sell_order)

                return

            # only asks exist: place bid below them, if room to buy
            if best_bid is None and best_ask is not None:
                bid_price = best_ask - 2 * tick

                if can_buy(ORDER_SIZE):
                    buy_order = OrderRequest(
                        product=self.product,
                        price=bid_price,
                        side=Side.BUY,
                        volume=ORDER_SIZE,
                    )
                    bot.send_order(buy_order)

                return

            # only bids exist: place ask above them, if room to sell
            if best_ask is None and best_bid is not None:
                ask_price = best_bid + 2 * tick

                if can_sell(ORDER_SIZE):
                    sell_order = OrderRequest(
                        product=self.product,
                        price=ask_price,
                        side=Side.SELL,
                        volume=ORDER_SIZE,
                    )
                    bot.send_order(sell_order)

                return

            # normal case: both sides exist
            mid = (best_bid + best_ask) / 2
            diff = predicted_value - mid

            # bullish signal: want to get longer, but only if within +200
            if diff > THRESHOLD and can_buy(ORDER_SIZE):
                bid_price = mid - tick
                buy_order = OrderRequest(
                    product=self.product,
                    price=bid_price,
                    side=Side.BUY,
                    volume=ORDER_SIZE,
                )
                bot.send_order(buy_order)
                return

            # bearish signal: want to get shorter, but only if within -200
            if diff < -THRESHOLD and can_sell(ORDER_SIZE):
                ask_price = mid + tick
                sell_order = OrderRequest(
                    product=self.product,
                    price=ask_price,
                    side=Side.SELL,
                    volume=ORDER_SIZE,
                )
                bot.send_order(sell_order)
                return

            # inside threshold or no inventory room: stay flat (no new orders)


class BavarianceBot(BaseBot):
    def __init__(self, cmi_url: str, username: str, password: str):
        super.__init__(cmi_url, username, password)

        self.product_managers = {}
        for product, model in PRODUCT_MODELS.items():
            self.product_managers[product] = ProductManager(product, model)

    # Handler for own trades
    def on_trades(self, trades: list[dict]):
        pass

    # Handler for order book updates
    def on_orderbook(self, orderbook: OrderBook):
        manager = self.product_managers.get(product)
        if manager is not None:
            manager.handle_orderbook(orderbook)
        else:
            # Product name error (KeyError), but this should never happen anyway
            pass

    def place_example_order(self):
        order = OrderRequest(
            product="SOME_SYMBOL",
            price=100.0,
            side=Side.BUY,
            volume=1,
        )
        resp = self.send_order(order)
        print("Order response:", resp)
