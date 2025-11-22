from base_bot import BaseBot
from imc_types import OrderBook, Trade, Side, OrderRequest

from prediction_model import PredictionModel


PRODUCTS = [
    '1_Eisbach',
    '2_Eisbach_Call',
    '3_Weather',
    '4_Weather',
    '5_Flights',
    '6_Airport',
    '7_ETF',
    '8_ETF_Strangle'
]


class ProductManager:

    def __init__(self, product: str, prediction_model: PredictionModel):
        self.product = product
        self.prediction_model = prediction_model
        self.realtime_flag = False

    @staticmethod
    def separate_orders():
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

    def handle_orderbook(self, orderbook: OrderBook):
        own_buy_orders, others_buy_orders, own_sell_orders, others_sell_orders = self.separate_orders()

        # TODO: Check if we have realtime data

        if realtime_flag:
            # TODO: Implement new realtime data logic

            realtime_flag = False
        else:
            # TODO: Implement old data prediction logic
            pass


class BavarianceBot(BaseBot):
    def __init__(self, cmi_url: str, username: str, password: str):
        super.__init__(cmi_url, username, password)

        self.product_managers = {}
        for product in PRODUCTS:
            self.product_managers[product] = ProductManager(product)

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
