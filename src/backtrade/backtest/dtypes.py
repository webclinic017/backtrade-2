from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar

import attrs
import fitter
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame, Series, Timedelta, Timestamp, concat
from pandas.core.groupby.generic import SeriesGroupBy
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
    name: str | None = attrs.field(on_setattr=attrs.setters.frozen)
    close: Series[float] | DataFrame = attrs.field(on_setattr=attrs.setters.frozen)
    position: Series[float] = attrs.field(on_setattr=attrs.setters.frozen)
    position_quote: Series[float] = attrs.field(on_setattr=attrs.setters.frozen)
    balance_quote: Series[float] = attrs.field(on_setattr=attrs.setters.frozen)
    equity_quote: Series[float] = attrs.field(on_setattr=attrs.setters.frozen)
    maker_fee_rate: float = attrs.field(on_setattr=attrs.setters.frozen)
    taker_fee_rate: float = attrs.field(on_setattr=attrs.setters.frozen)
    finished_orders: Series[FinishedOrder[_IndexType, Any]] = attrs.field(
        on_setattr=attrs.setters.frozen
    )

    logarithmic: bool = attrs.field(on_setattr=attrs.setters.NO_OP)
    freq: Timedelta = attrs.field(
        default=Timedelta("1d"), init=False, on_setattr=attrs.setters.NO_OP
    )
    _profit_memory: joblib.Memory = attrs.field(
        default=joblib.Memory(),
        init=False,
        on_setattr=attrs.setters.NO_OP,
        repr=False,
        eq=False,
        order=False,
        hash=False,
        metadata={"pickle": False},
    )
    _finished_orders_by_index_memory: joblib.Memory = attrs.field(
        default=joblib.Memory(),
        init=False,
        on_setattr=attrs.setters.frozen,
        repr=False,
        eq=False,
        order=False,
        hash=False,
        metadata={"pickle": False},
    )

    def __add__(self, other: BacktestResult[_IndexType]) -> BacktestResult[_IndexType]:
        if not isinstance(other, BacktestResult):
            return NotImplemented  # type: ignore
        if self.logarithmic != other.logarithmic:
            raise ValueError(
                "Cannot add backtest results with different logarithmic settings"
            )

        _TSeries = TypeVar("_TSeries", bound="Series")

        def reindex_fill(s1: _TSeries, s2: _TSeries) -> tuple[_TSeries, _TSeries]:
            new_index = s1.index.union(s2.index)
            s1 = s1.reindex(new_index).ffill().bfill()
            s2 = s2.reindex(new_index).ffill().bfill()
            return s1, s2

        def add_fbfill(s1: _TSeries, s2: _TSeries) -> _TSeries:
            s1, s2 = reindex_fill(s1, s2)
            return s1 + s2

        def add_fill0(s1: _TSeries, s2: _TSeries) -> _TSeries:
            return s1.add(s2, fill_value=0)

        close_1 = self.close
        close_2 = other.close
        close_concat_axis = int(not close_1.index.intersection(close_2.index).empty)
        if close_concat_axis == 1:
            # Rename if Series

            if isinstance(close_1, Series):
                close_1.rename(f"{self.name}_Close", inplace=True)
            if isinstance(close_2, Series):
                close_2.rename(f"{other.name}_Close", inplace=True)

            # Concat

            new_close = (
                concat([close_1, close_2], axis=close_concat_axis).ffill().bfill()
            )

            # Avoid duplicate columns
            def rename_duplicated(df: DataFrame) -> None:
                from pandas.io.parsers.base_parser import ParserBase

                df.columns = ParserBase({"usecols": None})._maybe_dedup_names(
                    df.columns
                )

            rename_duplicated(new_close)
        else:
            new_close = concat([close_1, close_2], axis=close_concat_axis)

        balance_quote_1 = self.balance_quote
        balance_quote_2 = other.balance_quote
        balance_quote_1.iat[-1] = self.equity_quote.iat[-1]
        balance_quote_2.iat[-1] = other.equity_quote.iat[-1]
        balance_quote_1.iat[0] = self.equity_quote.iat[0]
        balance_quote_2.iat[0] = other.equity_quote.iat[0]
        return BacktestResult(
            name=f"{self.name} + {other.name}",
            close=new_close,
            position=add_fill0(self.position, other.position),
            position_quote=add_fill0(self.position_quote, other.position_quote),
            balance_quote=add_fbfill(balance_quote_1, balance_quote_2),
            equity_quote=add_fbfill(self.equity_quote, other.equity_quote),
            maker_fee_rate=self.maker_fee_rate
            if self.maker_fee_rate == other.maker_fee_rate
            else np.nan,
            taker_fee_rate=self.taker_fee_rate
            if self.taker_fee_rate == other.taker_fee_rate
            else np.nan,
            finished_orders=concat(
                [self.finished_orders, other.finished_orders], sort=True
            ),
            logarithmic=self.logarithmic,
        )

    def __mul__(self, other: float) -> BacktestResult[_IndexType]:
        if not isinstance(other, float):
            return NotImplemented  # type: ignore
        if other < 0:
            raise ValueError(f"Cannot multiply by negative number: {other}")

        return attrs.evolve(
            self,
            position=self.position * other,
            position_quote=self.position_quote * other,
            balance_quote=self.balance_quote * other,
            equity_quote=self.equity_quote * other,
            finished_orders=self.finished_orders * other,
        )

    def __div__(self, other: float) -> BacktestResult[_IndexType]:
        return self * (1 / other)

    def _finished_orders_by_index(
        self,
    ) -> SeriesGroupBy[FinishedOrder[_IndexType @ BacktestResult, Any]]:
        return self.finished_orders.groupby(level=0)

    @property
    def finished_orders_by_index(
        self,
    ) -> SeriesGroupBy[FinishedOrder[_IndexType @ BacktestResult, Any]]:
        return self._finished_orders_by_index_memory.cache(
            self._finished_orders_by_index
        )()

    @property
    def order_count(self) -> Series[int]:
        return self.finished_orders_by_index.count().reindex(
            self.close.index, fill_value=0
        )

    @property
    def filled_rate(self) -> Series[float]:
        return self.finished_orders_by_index.apply(
            lambda series: series.apply(lambda order: order.filled).mean()
        ).reindex(self.close.index)

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

    def _profit(self) -> Series[float]:
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
    def profit(self) -> Series[float]:
        return self._profit_memory.cache(self._profit)()

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
    def total_order_amount(self) -> float:
        return self.finished_orders.apply(lambda x: abs(x.quote_size)).sum()

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
    def win_ratio(self) -> float:
        return self.profit.apply(lambda x: 1 if x > 0 else 0).mean()

    @property
    def _all_metrics(self) -> Series:
        s = Series(
            {
                "Frequency (User Specified)": self.freq,
                "Logarithmic (User Specified)": self.logarithmic,
                "Period": self.period,
                "Win Ratio": self.win_ratio,
                "Profit Average": self.profit.mean(),
                "Profit Median": self.profit.median(),
                "Profit Std": self.profit.std(),
                "Profit Skewness": self.profit.skew(),
                "Profit Kurtosis": self.profit.kurt(),
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
                "Total Order Amount": self.total_order_amount,
                "Total Orders Count": self.total_orders_count,
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

    def plot(self, *, use_fitter: bool = False) -> Figure:
        # Create Subfigures
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        fig.suptitle(f"Backtest for {self.name}", fontsize=16)
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
                self.close
                if isinstance(self.close, Series)
                else self.close.div(self.close.max(axis=0), axis=1),
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
                self.close.columns
                if isinstance(self.close, DataFrame)
                else [self.close.name],
                ["Position"],
                ["Position (Quote)"],
                ["Balance (Quote)"],
                ["Equity (Quote)"],
                df.columns[df.columns.str.contains("Filled Rate")],
                df.columns[df.columns.str.contains("Order Count")],
            ],
            ax=axes,
            kind="line",
        )
        if self.logarithmic:
            for i in [1, 2, 3, 4]:
                axes[i].set_yscale("log")

        # Second Subfigure
        axes = subfigs[1].subplots(
            5 if use_fitter else 4,
            1,
            gridspec_kw={
                "height_ratios": [1.95, 0.6, 0.5, 1, 1]
                if use_fitter
                else [1.95, 0.6, 1, 1]
            },
        )

        df.index.name = "DateTime"
        df = df.loc[:, ~df.columns.str.contains("SMA")]
        df.columns = df.columns.str.replace(" ", "\n")

        # Metrics
        Table(self._all_metrics.to_frame(), ax=axes[0])

        if use_fitter:
            plt.sca(axes[1])
            fitter_ = fitter.Fitter(
                self.profit.dropna(), distributions=fitter.get_common_distributions()
            )
            fitter_.fit()
            fit_summary = (
                fitter_.summary(plot=True, clf=False)
                .sort_values("sumsquare_error")[["sumsquare_error", "ks_pvalue"]]
                .round(3)
            )
            Table(fit_summary, ax=axes[2], textprops={"size": 8})
        else:
            self.profit.plot(ax=axes[1], kind="hist", bins=100, title="Profit")

        # Trades
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
            ax=axes[3 if use_fitter else 2],
            column_definitions=coldefs,
            textprops=textprops,
        )
        Table(
            df.tail(10).round(2),
            ax=axes[4 if use_fitter else 3],
            column_definitions=coldefs,
            textprops=textprops,
        )

        return fig

    @property
    def flatten_finished_orders(self) -> Series[FinishedOrder[_IndexType, Any]]:
        return self.finished_orders.explode().dropna()
