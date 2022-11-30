from datetime import datetime, timedelta
from typing import Any, Generic

import attrs
import joblib
import pandas as pd
import pandas_ta as ta
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import Series, Timedelta, Timestamp, concat
from plottable import ColDef, Table

from backtrade.logic import FinishedOrder, FinishedOrderState

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


@attrs.define(kw_only=True)
class BacktestResult(Generic[_IndexType]):
    close: "Series[float]" = attrs.field(on_setattr=attrs.setters.frozen)
    position: "Series[float]" = attrs.field(on_setattr=attrs.setters.frozen)
    position_quote: "Series[float]" = attrs.field(on_setattr=attrs.setters.frozen)
    balance_quote: "Series[float]" = attrs.field(on_setattr=attrs.setters.frozen)
    equity_quote: "Series[float]" = attrs.field(on_setattr=attrs.setters.frozen)
    filled_rate: "Series[float]" = attrs.field(on_setattr=attrs.setters.frozen)
    maker_fee_rate: float = attrs.field(on_setattr=attrs.setters.frozen)
    taker_fee_rate: float = attrs.field(on_setattr=attrs.setters.frozen)
    finished_orders: "Series[FinishedOrder[_IndexType, Any]]" = attrs.field(
        on_setattr=attrs.setters.frozen
    )
    order_count: "Series[int]" = attrs.field(on_setattr=attrs.setters.frozen)

    logarithmic: bool = attrs.field(
        default=True, init=False, on_setattr=attrs.setters.NO_OP
    )
    freq: Timedelta = attrs.field(
        default=Timedelta("1d"), init=False, on_setattr=attrs.setters.NO_OP
    )
    memory: joblib.Memory = attrs.field(
        default=joblib.Memory(), init=False, on_setattr=attrs.setters.NO_OP
    )

    @property
    def period(self) -> Timedelta:
        first_index = self.close.index[0]
        last_index = self.close.index[-1]

        if isinstance(first_index, Timestamp) and isinstance(last_index, Timestamp):
            return last_index - first_index
        elif isinstance(first_index, Timedelta) and isinstance(last_index, Timedelta):
            return last_index - first_index
        elif isinstance(first_index, datetime) and isinstance(last_index, datetime):
            return Timedelta(last_index - first_index)
        elif isinstance(first_index, timedelta) and isinstance(last_index, timedelta):
            return Timedelta(last_index - first_index)
        else:
            raise TypeError("Index must be Timestamp or Timedelta")

    def _profit(self) -> "Series[float]":
        equity_resampled = (
            self.equity_quote.resample(self.freq)
            .first()
            .fillna(method="ffill")
            .dropna()
        )
        if self.logarithmic:
            profit = equity_resampled.pct_change() - 1
        else:
            profit = equity_resampled.diff()
        return profit

    @property
    def profit(self) -> "Series[float]":
        return self.memory.cache(self._profit)()

    @property
    def annual_sharp_ratio(self) -> float:
        return self.profit.mean() / self.annual_volatility

    @property
    def annual_sortino_ratio(self) -> float:
        return (
            self.profit.mean()
            / self.profit[self.profit < 0].std()
            * (Timedelta(days=365) / self.period) ** 0.5
        )

    @property
    def max_drawdown(self) -> float:
        if self.logarithmic:
            return ta.max_drawdown(self.equity_quote, method="percent")
        else:
            return (
                ta.max_drawdown(self.equity_quote, method="percent")
                * self.equity_quote.iat[0]
            )

    @property
    def annual_volatility(self) -> float:
        return self.profit.std() * (Timedelta(days=365) / self.period) ** 0.5

    @property
    def total_fee(self) -> float:
        return self.finished_orders.apply(lambda x: x.fee).sum()

    @property
    def total_maker_fee(self) -> float:
        return self.finished_orders.apply(
            lambda x: x.fee if x.state == FinishedOrderState.FilledMaker else 0
        ).sum()

    @property
    def total_taker_fee(self) -> float:
        return self.finished_orders.apply(
            lambda x: x.fee if x.state == FinishedOrderState.FilledTaker else 0
        ).sum()

    @property
    def fee_ratio(self) -> float:
        return self.total_fee / (
            self.equity_quote.iat[-1] - self.equity_quote.iat[0] + self.total_fee
        )

    @property
    def total_orders_count(self) -> int:
        return self.finished_orders.size

    @property
    def state_maker_ratio(self) -> float:
        return self.finished_orders.apply(
            lambda x: 1 if x.state == FinishedOrderState.FilledMaker else 0
        ).mean()

    @property
    def state_taker_ratio(self) -> float:
        return self.finished_orders.apply(
            lambda x: 1 if x.state == FinishedOrderState.FilledTaker else 0
        ).mean()

    @property
    def state_cancelled_not_filled_ratio(self) -> float:
        return self.finished_orders.apply(
            lambda x: 1 if x.state == FinishedOrderState.CancelledNotFilled else 0
        ).mean()

    @property
    def state_cancelled_post_only_ratio(self) -> float:
        return self.finished_orders.apply(
            lambda x: 1 if x.state == FinishedOrderState.CancelledPostOnly else 0
        ).mean()

    @property
    def _all_metrics(self) -> Series:
        s = Series(
            {
                "Frequency (User Specified)": self.freq,
                "Logarithmic (User Specified)": self.logarithmic,
                "Period": self.period,
                "Annual Volatility": self.annual_volatility,
                "Annual Sharp Ratio": self.annual_sharp_ratio,
                "Annual Sortino Ratio": self.annual_sortino_ratio,
                "Max Drawdown": f"{self.max_drawdown:.3%}",
                "Maker Fee Rate (User Specified)": f"{self.maker_fee_rate:.5%}",
                "Taker Fee Rate (User Specified)": f"{self.taker_fee_rate:.5%}",
                "Total Fee": self.total_fee,
                "Total Maker Fee": self.total_maker_fee,
                "Total Taker Fee": self.total_taker_fee,
                "Total Fee / Total Profit without Fee": f"{self.fee_ratio:.3%}",
                "State: Maker": f"{self.state_maker_ratio:.3%}"
                if self.total_orders_count > 0
                else "N/A",
                "State: Taker": f"{self.state_taker_ratio:.3%}"
                if self.total_orders_count > 0
                else "N/A",
                "State: Cancelled (Not Filled)"
                + "": f"{self.state_cancelled_not_filled_ratio:.3%}"
                if self.total_orders_count > 0
                else "N/A",
                "State: Cancelled (Post Only)"
                + "": f"{self.state_cancelled_post_only_ratio:.3%}"
                if self.total_orders_count > 0
                else "N/A",
            },
            name="Value",
        ).rename_axis("Metric Name")
        s[s.apply(lambda x: isinstance(x, float))] = s[
            s.apply(lambda x: isinstance(x, float))
        ].apply(lambda x: f"{x:.3f}")
        return s

    def plot(self) -> Figure:
        # Create Subfigures
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        subfigs: list[Figure] = fig.subfigures(1, 2, width_ratios=[3, 1.3])

        # First Subfigure
        axes = subfigs[0].subplots(7, 1, sharex=True)
        data_len = len(self.close)
        sma_len = data_len // 20

        filled_rate_sma = ta.sma(self.filled_rate.dropna(), length=sma_len)
        if filled_rate_sma is not None:
            filled_rate_sma.rename("Filled Rate SMA", inplace=True)
        else:
            filled_rate_sma = Series(
                name="Filled Rate SMA (N/A, Not Enough Data)", dtype=float
            )
        order_count_sma = ta.sma(self.order_count.dropna(), length=sma_len)
        if order_count_sma is not None:
            order_count_sma.rename("Order Count SMA", inplace=True)
        else:
            order_count_sma = Series(
                name="Order Count SMA (N/A, Not Enough Data)", dtype=float
            )
        df = concat(
            [
                self.close.rename("Close"),
                self.position.rename("Position"),
                self.position_quote.rename("Position (Quote)"),
                self.balance_quote.rename("Balance (Quote)"),
                self.equity_quote.rename("Equity (Quote)"),
                self.filled_rate.rename("Filled Rate"),
                filled_rate_sma,
                self.order_count.rename("Order Count"),
                order_count_sma,
            ],
            axis=1,
        )
        df.plot(
            subplots=[
                ["Close"],
                ["Position"],
                ["Position (Quote)"],
                ["Balance (Quote)"],
                ["Equity (Quote)"],
                df.columns[df.columns.str.contains("Filled Rate")],
                df.columns[df.columns.str.contains("Order Count")],
            ],
            ax=axes,
        )

        # Second Subfigure
        axes = subfigs[1].subplots(3, 1, gridspec_kw={"height_ratios": [1.3, 1, 1]})

        df.index.name = "DateTime"
        df = df.loc[:, ~df.columns.str.contains("SMA")]
        df.columns = df.columns.str.replace(" ", "\n")

        Table(self._all_metrics.to_frame(), ax=axes[0])
        coldefs = [
            ColDef(
                "DateTime",
                formatter=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"),
                textprops={"size": 8},
            )
        ]
        textprops = {"size": 9.5}
        Table(
            df.head(10).round(2),
            ax=axes[1],
            column_definitions=coldefs,
            textprops=textprops,
        )
        Table(
            df.tail(10).round(2),
            ax=axes[2],
            column_definitions=coldefs,
            textprops=textprops,
        )

        return fig

    @property
    def flatten_finished_orders(self) -> "Series[FinishedOrder[_IndexType, Any]]":
        return self.finished_orders.explode().dropna()
