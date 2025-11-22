from dataclasses import dataclass, asdict
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, Any

class DictLikeFrozenDataclassMapping(Mapping):
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__annotations__)

    def __len__(self) -> int:
        return len(self.__annotations__)

    def to_dict(self) -> dict:
        return asdict(self)

    def keys(self):
        return self.__annotations__.keys()

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]


@dataclass(frozen=True)
class Product(DictLikeFrozenDataclassMapping):
    symbol: str
    tickSize: float
    startingPrice: int
    contractSize: int


@dataclass(frozen=True)
class Trade(DictLikeFrozenDataclassMapping):
    timestamp: str
    product: str
    buyer: str
    seller: str
    volume: int
    price: float


@dataclass(frozen=True)
class Order(DictLikeFrozenDataclassMapping):
    price: float
    volume: int
    own_volume: int


@dataclass(frozen=True)
class OrderBook(DictLikeFrozenDataclassMapping):
    product: str
    tick_size: float
    buy_orders: list[Order]
    sell_orders: list[Order]


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderRequest:
    product: str
    price: float
    side: Side
    volume: int


@dataclass(frozen=True)
class OrderResponse:
    id: str
    status: Literal["ACTIVE", "PART_FILLED"]
    product: str
    side: Side
    price: float
    volume: int
    filled: int
    user: str
    timestamp: str
    targetUser: str | None
    message: str | None
