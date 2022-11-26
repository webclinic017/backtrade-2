from typing import Any, Generic

import attrs
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import Series, concat

from backtrade.logic import FinishedOrder

from ..order import _IndexType

__all__ = ["BacktestResult", "CloseData"]


@attrs.frozen(kw_only=True)
class CloseData(Generic[_IndexType]):
    index: _IndexType
    open: float
    high: float
    low: float
    close: float
    position: float
    position_quote: float
    balance_quote: float
    equity_quote: float


@attrs.frozen(kw_only=True)
class BacktestResult(Generic[_IndexType]):
    close: "Series[float]"
    position: "Series[float]"
    position_quote: "Series[float]"
    balance_quote: "Series[float]"
    equity_quote: "Series[float]"
    filled_rate: "Series[float]"
    finished_orders: "Series[FinishedOrder[_IndexType, Any]]"
    order_count: "Series[int]"

    def plot(self) -> Figure:
        fig = plt.figure()
        df = concat(
            [
                self.close,
                self.position,
                self.position_quote,
                self.balance_quote,
                self.equity_quote,
                self.filled_rate,
                self.order_count,
            ],
            axis=1,
        )
        df.plot(subplots=True, sharex=True)
        return fig

    @property
    def flatten_finished_orders(self) -> "Series[FinishedOrder[_IndexType, Any]]":
        return self.finished_orders.explode().dropna()
