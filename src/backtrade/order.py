from abc import ABCMeta
from typing import Hashable, TypeVar, Union

import attrs


@attrs.frozen(kw_only=True)
class OrderBase(metaclass=ABCMeta):
    size: float


@attrs.frozen(kw_only=True)
class LimitOrder(OrderBase):
    price: float
    post_only: bool


@attrs.frozen(kw_only=True)
class MarketOrder(OrderBase):
    pass


_IndexType = TypeVar("_IndexType", bound=Hashable)
_OrderType = TypeVar("_OrderType", bound=Union[LimitOrder, MarketOrder])
