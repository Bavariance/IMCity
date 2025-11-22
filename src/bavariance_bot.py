from base_bot import BaseBot
from imc_types import OrderBook, Trade, Side, OrderRequest


class BavarianceBot(BaseBot):
    def on_orderbook(self, orderbook: OrderBook):
        # TODO
        pass

    def on_trades(self, trades: list[Trade]):
        # TODO
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
