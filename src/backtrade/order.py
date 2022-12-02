from abc import ABCMeta
from typing import Any, Generic, Hashable, TypeVar, Union

import attrs

from .validator import _na_validator

_TOrder = TypeVar("_TOrder", bound="OrderBase[Any]")


@attrs.frozen(kw_only=True)
class OrderBase(Generic[_TOrder], metaclass=ABCMeta):
    size: float = attrs.field(validator=_na_validator)

    def __mul__(self: "_TOrder", other: float) -> "_TOrder":
        if not isinstance(other, float):
            return NotImplemented  # type: ignore
        if other < 0:
            raise ValueError(f"Cannot multiply by negative number: {other}")
        return attrs.evolve(self, size=self.size * other)

    def __div__(self: "_TOrder", other: float) -> "_TOrder":
        return self * (1 / other)


@attrs.frozen(kw_only=True)
class LimitOrder(OrderBase["LimitOrder"]):
    price: float = attrs.field(validator=_na_validator)
    post_only: bool = attrs.field(validator=_na_validator)


@attrs.frozen(kw_only=True)
class MarketOrder(OrderBase["MarketOrder"]):
    pass


_IndexType = TypeVar("_IndexType", bound=Hashable)
_OrderType = TypeVar("_OrderType", bound=Union[LimitOrder, MarketOrder])
