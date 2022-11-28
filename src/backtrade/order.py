from abc import ABCMeta
from typing import Hashable, TypeVar, Union

import attrs

from .validator import _na_validator


@attrs.frozen(kw_only=True)
class OrderBase(metaclass=ABCMeta):
    size: float = attrs.field(validator=_na_validator)


@attrs.frozen(kw_only=True)
class LimitOrder(OrderBase):
    price: float = attrs.field(validator=_na_validator)
    post_only: bool = attrs.field(validator=_na_validator)


@attrs.frozen(kw_only=True)
class MarketOrder(OrderBase):
    pass


_IndexType = TypeVar("_IndexType", bound=Hashable)
_OrderType = TypeVar("_OrderType", bound=Union[LimitOrder, MarketOrder])
