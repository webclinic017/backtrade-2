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
    fee: float
    state: FinishedOrderState

    @property
    def filled(self) -> bool:
        return self.state in (
            FinishedOrderState.FilledTaker,
            FinishedOrderState.FilledMaker,
        )
