from __future__ import annotations

from enum import Enum
from typing import Generic

import attrs

from .order import _IndexType, _OrderType


class FinishedOrderState(Enum):
    FilledTaker = 0
    FilledMaker = 1
    CancelledNotFilled = 2
    CancelledPostOnly = 3


@attrs.frozen(kw_only=True)
class FinishedOrder(Generic[_IndexType, _OrderType]):
    index: _IndexType
    order: _OrderType
    balance_decrement: float
    executed_price: float | None
    quote_size: float
    fee: float
    state: FinishedOrderState

    @property
    def filled(self) -> bool:
        return self.state in (
            FinishedOrderState.FilledTaker,
            FinishedOrderState.FilledMaker,
        )

    def __mul__(self, other: float) -> FinishedOrder[_IndexType, _OrderType]:
        if not isinstance(other, float):
            return NotImplemented  # type: ignore
        if other < 0:
            raise ValueError(f"Cannot multiply by negative number: {other}")
        return attrs.evolve(
            self,
            order=self.order * other,
            balance_decrement=self.balance_decrement * other,
            quote_size=self.quote_size * other,
            fee=self.fee * other,
        )

    def __div__(self, other: float) -> FinishedOrder[_IndexType, _OrderType]:
        return self * (1 / other)
