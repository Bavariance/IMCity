import json
from threading import Thread
from typing import Any, Callable

import requests
import sseclient

from imc_types import OrderBook, Order

STANDARD_HEADERS = {"Content-Type": "application/json; charset=utf-8"}


class SSEThread(Thread):
    def __init__(
        self,
        bearer: str,
        url: str,
        handle_orderbook: Callable[[OrderBook], Any],
        handle_trade_event: Callable[[dict], Any],
    ):
        super().__init__()
        self.bearer = bearer
        self.url = url
        self._handle_orderbook = handle_orderbook
        self._handle_trade_event = handle_trade_event
        self._http_stream = None
        self._client = None
        self._closed = False

    def run(self):
        while not self._closed:
            try:
                self._start_sse_client()
            except Exception:
                if not self._closed:
                    print("Encountered an error. Trying to restart SSE client...")
                    import traceback
                    print(traceback.format_exc())

    def close(self):
        self._closed = True
        if self._http_stream:
            self._http_stream.close()
        if self._client:
            self._client.close()

    def _handle_orderbook_change(self, orderbook: dict[str, Any]):
        buy_orders = sorted(
            [
                {
                    "price": float(price),
                    "volume": volumes["marketVolume"],
                    "own_volume": volumes["userVolume"],
                }
                for price, volumes in orderbook["buyOrders"].items()
            ],
            key=lambda d: -d["price"],
        )
        sell_orders = sorted(
            [
                {
                    "price": float(price),
                    "volume": volumes["marketVolume"],
                    "own_volume": volumes["userVolume"],
                }
                for price, volumes in orderbook["sellOrders"].items()
            ],
            key=lambda d: d["price"],
        )

        self._handle_orderbook(
            OrderBook(
                orderbook["productsymbol"],
                orderbook["tickSize"],
                [Order(**o) for o in buy_orders],
                [Order(**o) for o in sell_orders],
            )
        )

    def _start_sse_client(self):
        headers = {
            "Authorization": self.bearer,
            "Accept": "text/event-stream; charset=utf-8",
        }
        self._http_stream = requests.get(
            self.url, stream=True, headers=headers, timeout=30
        )
        self._client = sseclient.SSEClient(self._http_stream)

        for event in self._client.events():
            if event.event == "order":
                self._handle_orderbook_change(json.loads(event.data))
            elif event.event == "trade":
                self._handle_trade_event(json.loads(event.data))

