from base_bot import BaseBot
from imc_types import OrderBook, Trade, Side, OrderRequest


class BavarianceBot(BaseBot):
    # Handler for own trades
    def on_trades(self, trades: list[dict]):
        for trade in trades:
            print(f"{trade['volume']} @ {trade['price']}")

    # Handler for order book updates
    def on_orderbook(self, orderbook: OrderBook):
        print(orderbook)
        do_i_want_to_buy = False
        if do_i_want_to_buy:
            order = OrderRequest(product=orderbook.product,
                                 price=orderbook.sell_orders[0].price,
                                 volume=1,
                                 side=Side.BUY)
            self.send_order(order)

    def place_example_order(self):
        order = OrderRequest(
            product="SOME_SYMBOL",
            price=100.0,
            side=Side.BUY,
            volume=1,
        )
        resp = self.send_order(order)
        print("Order response:", resp)
