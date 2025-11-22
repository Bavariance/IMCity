# bot_base.py
import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from functools import cached_property
from typing import Any, Callable

import requests

from imc_types import (
    Product,
    OrderBook,
    Trade,
    OrderRequest,
    OrderResponse,
)
from sse_thread import SSEThread, STANDARD_HEADERS


class BaseBot(ABC):
    username: str
    _password: str
    _cmi_url: str
    _sse_thread: SSEThread = None

    def __init__(self, cmi_url: str, username: str, password: str):
        self._cmi_url = cmi_url
        self.username = username
        self._password = password

    @cached_property
    def auth_token(self):
        return self._authenticate()

    def start(
        self, on_orderbook: Callable | None = None, on_trades: Callable | None = None
    ) -> None:
        """
        Creates SSE thread to handle market events
        """
        if self._sse_thread:
            raise Exception(
                "Bot already running. Please use the `stop()` method before trying again."
            )

        self._sse_thread = SSEThread(
            bearer=self.auth_token,
            url=f"{self._cmi_url}/api/market/stream",
            handle_orderbook=on_orderbook or self.on_orderbook,
            handle_trade_event=on_trades or self.on_trades,
        )

        print("Starting SSEThread...")
        self._sse_thread.start()
        print("SSEThread started.")

    def stop(self) -> None:
        """
        Closes SSE thread
        """
        print("Closing SSE Thread...")
        self._sse_thread.close()
        self._sse_thread.join()
        self._sse_thread = None
        print("SSE Thread closed")

    @abstractmethod
    def on_orderbook(self, orderbook: OrderBook):
        raise NotImplementedError("You must implement the on_orderbook method!")

    @abstractmethod
    def on_trades(self, trades: list[Trade]):
        raise NotImplementedError("You must implement the on_trades method!")

    def _get_headers(self) -> dict[str, str]:
        return {**STANDARD_HEADERS, "Authorization": self.auth_token}

    def send_order(self, order_request: OrderRequest) -> OrderResponse | None:
        payload = asdict(order_request)
        url = f"{self._cmi_url}/api/order"
        response = requests.post(url, json=payload, headers=self._get_headers())
        if response.status_code == 200:
            return OrderResponse(**response.json())
        else:
            print(
                f"Failed to send order, {order_request}, with response {response.content}"
            )

    def send_mass_orders(
        self, order_requests: list[OrderRequest]
    ) -> list[OrderResponse]:
        responses = []

        def worker(order_request, response_list):
            response = self.send_order(order_request)
            response_list.append(response)

        threads = []
        for order_request in order_requests:
            thread = Thread(target=worker, args=(order_request, responses))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return responses

    def request_all_orders(self) -> list[dict] | None:
        url = f"{self._cmi_url}/api/order/current-user"
        response = requests.get(url, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get all orders: {response.content}")

    def cancel_order_by_id(self, order_id: str) -> dict | None:
        url = f"{self._cmi_url}/api/order/{order_id}"
        response = requests.delete(url, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()

        print(f"Failed to cancel order: {response.content}")

    def cancel_order(self, product: str, price: float) -> dict | None:
        url = f"{self._cmi_url}/api/order?product={product}&price={price}"
        response = requests.delete(url, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to cancel order: {response.content}")

    def cancel_all_orders(self) -> None:
        for order in self.request_all_orders():
            url = f"{self._cmi_url}/api/order/{order['id']}"
            response = requests.delete(url, headers=self._get_headers())
            if response.status_code != 200:
                print(f"Failed to cancel order: {response.content}")

    def request_all_products(self) -> list[Product] | None:
        url = f"{self._cmi_url}/api/product"
        response = requests.get(url, headers=self._get_headers())
        if response.status_code == 200:
            return list(map(lambda prod: Product(**prod), json.loads(response.text)))
        else:
            print(f"Failed to get all products: {response.content}")

    def request_positions(self) -> dict[str, int] | None:
        url = f"{self._cmi_url}/api/position/current-user"
        response = requests.get(url, headers=self._get_headers())
        if response.status_code == 200:
            return {
                position["product"]: position["volume"] for position in response.json()
            }
        else:
            print(f"Failed to get positions: {response.content}")

    def request_net_positions(self) -> dict[str, int] | None:
        url = f"{self._cmi_url}/api/position/current-user"
        response = requests.get(url, headers=self._get_headers())
        if response.status_code == 200:
            return {
                position["product"]: position["netPosition"]
                for position in response.json()
            }
        else:
            print(f"Failed to get net positions for user: {response.content}")

    def _authenticate(self) -> str:
        auth = {"username": self.username, "password": self._password}
        url = f"{self._cmi_url}/api/user/authenticate"
        response = requests.post(url, headers=STANDARD_HEADERS, json=auth)
        response.raise_for_status()

        return response.headers["Authorization"]
