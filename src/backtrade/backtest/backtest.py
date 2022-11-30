from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from logging import getLogger
from typing import Any, Generic, Iterable, final

import numpy as np
from exceptiongroup import ExceptionGroup
from pandas import DataFrame, Series

from backtrade.logic import FinishedOrder, ProcessOrderArgs, process_order

from ..order import LimitOrder, MarketOrder, _IndexType
from .dtypes import BacktestResult, CloseData

__all__ = ["Backtester"]


class Backtester(Generic[_IndexType], metaclass=ABCMeta):
    """Simple backtester.

    Supports hedge mode, different `maker` / `taker` fees,
    `market`, `limit`, `post-only limit` orders.

    Every unfilled orders are to be automatically canceled
    on the next time step.

    Example
    ----------
    >>> import pandas as pd
    >>> from backtrade import (Backtester, CloseData, LimitOrder,
            MarketOrder, _IndexType)
    >>> df = pd.concat([open, close, high, low, signal, ATR], axis=1)
    >>> class MyBacktester(Backtester[_IndexType]):
    >>>    def init(self):
    >>>        pass
    >>>    def on_close(
                self,
                close_data: "CloseData[_IndexType]",
                row: "Series[Any]"
            ) -> "Iterable[Union[MarketOrder, LimitOrder]]":
    >>>        if row["signal"] == 1:
    >>>            yield Order(size=1, price=row['close'] - row['ATR'] * 0.5,
                               limit=True, post_only=True)
    >>>        if row["signal"] == -1:
    >>>            yield Order(size=-1, price=row['close'] + row['ATR'] * 0.5,
                               limit=True, post_only=True)
    >>> bt = Backtester()
    >>> bt(df, maker_fee=-0.01 * 0.025, taker_fee=0.01 * 0.075).plot()

    """

    @final
    def __call__(
        self,
        df: DataFrame,
        *,
        maker_fee: float,
        taker_fee: float,
        balance_init: float = 1,
    ) -> BacktestResult[_IndexType]:
        """Initialize the backtester.

        Parameters
        ----------
        df : DataFrame
            Must have columns 'open', 'close', 'high', 'low'.
            df.loc[index, :] will be passed to `next` method.
        maker_fee : float
            Maker fee.
        taker_fee : float
            Taker fee.
        balance_init : float, optional
            Initial balance, by default 1
        """

        # Check arguments
        # Errors
        errors: list[ValueError] = []
        if not all(col in df.columns for col in ["open", "close", "high", "low"]):
            if all(col in df.columns for col in ["Open", "Close", "High", "Low"]):
                df = df.copy()
                df[["open", "close", "high", "low"]] = df[
                    ["Open", "Close", "High", "Low"]
                ]
            else:
                errors.append(
                    ValueError(
                        'df must have columns "open", "close", "high", "low", '
                        + f"but got {df.columns}"
                    )
                )
        if not errors:
            if (df["open"] > df["high"]).any():
                errors.append(ValueError("open price must be less than high price"))
            if (df["open"] < df["low"]).any():
                errors.append(ValueError("open price must be greater than low price"))
            if (df["close"] > df["high"]).any():
                errors.append(ValueError("close price must be less than high price"))
            if (df["close"] < df["low"]).any():
                errors.append(ValueError("close price must be greater than low price"))
            if (df["high"] < df["low"]).any():
                errors.append(ValueError("high price must be greater than low price"))
            if (df["open"] <= 0).any():
                errors.append(ValueError("open price must be greater than 0"))
            if (df["close"] <= 0).any():
                errors.append(ValueError("close price must be greater than 0"))
            if (df["high"] <= 0).any():
                errors.append(ValueError("high price must be greater than 0"))
            if (df["low"] <= 0).any():
                errors.append(ValueError("low price must be greater than 0"))
        if not df.index.is_monotonic_increasing:
            errors.append(ValueError("index must be monotonic increasing"))
        if not df.index.is_unique:
            errors.append(ValueError("index must be unique"))
        if not balance_init > 0:
            errors.append(ValueError("balance_init must be greater than 0"))

        # Warnings
        if not taker_fee > 0:
            warnings.warn(f"taker_fee is not positive (got {taker_fee}), are you sure?")

        # Raise errors
        if errors:
            raise ExceptionGroup(
                "Invalid arguments" + str([e.args[0] for e in errors]), errors
            )

        df = df.copy()
        # logger
        self.logger = getLogger(__name__)

        # Initialize
        self.init()

        # Calculate
        open_orders: tuple[LimitOrder | MarketOrder, ...] = ()
        last_close: float | None = None
        position = 0.0
        balance = balance_init

        position_history: Series[float] = Series(np.nan, index=df.index)
        position_quote_history: Series[float] = Series(np.nan, index=df.index)
        balance_quote_history: Series[float] = Series(np.nan, index=df.index)
        equity_quote_history: Series[float] = Series(np.nan, index=df.index)
        finished_orders_history: Series[list[FinishedOrder[_IndexType, Any]]] = Series(
            np.nan, index=df.index, dtype=object
        )

        for index, row in df.iterrows():
            index: _IndexType  # type: ignore
            row: Series[float]  # type: ignore
            open_ = row["open"]
            high = row["high"]
            low = row["low"]

            # Assert
            # assert open_ >= low
            # assert open_ <= high
            # assert low > 0
            # assert open_ > 0
            # assert high > 0

            # Iterate each open orders
            finished_orders: list[FinishedOrder[_IndexType, Any]] = []
            for order in open_orders:
                assert last_close is not None  # nosec
                if order.size == 0.0:
                    continue
                finished_order = process_order(
                    ProcessOrderArgs(
                        order=order,
                        last_close=last_close,
                        high=high,
                        low=low,
                        maker_fee=maker_fee,
                        taker_fee=taker_fee,
                        index=index,
                    )
                )
                balance -= finished_order.balance_decrement
                if finished_order.filled:
                    position += order.size
                finished_orders.append(finished_order)

            # Call at close
            equity = balance + position * open_
            position_quote = position * open_
            open_orders = tuple(
                self.on_close(
                    CloseData(
                        index=index,
                        open=open_,
                        high=high,
                        low=low,
                        close=row["close"],
                        position=position,
                        position_quote=position_quote,
                        balance_quote=balance,
                        equity_quote=equity,
                    ),
                    row,
                )
            )
            last_close = row["close"]

            position_history[index] = position
            position_quote_history[index] = position_quote
            balance_quote_history[index] = balance
            equity_quote_history[index] = equity
            finished_orders_history[index] = finished_orders

        order_count = finished_orders_history.apply(len)
        filled_rate: Series[float] = finished_orders_history.apply(
            lambda x: sum(1 for o in x if o.filled) / len(x) if x else np.nan
        )
        finished_orders_history_ = finished_orders_history.explode().dropna()

        return BacktestResult(
            close=df["close"],
            position=position_history.rename("position"),
            position_quote=position_quote_history.rename("position_quote"),
            balance_quote=balance_quote_history.rename("balance_quote"),
            equity_quote=equity_quote_history.rename("equity_quote"),
            finished_orders=finished_orders_history_.rename("finished_orders"),
            filled_rate=filled_rate.rename("filled_rate"),
            order_count=order_count.rename("order_count"),
            maker_fee_rate=maker_fee,
            taker_fee_rate=taker_fee,
        )

    def init(self) -> None:
        """If you want to initialize something, override this method.
        Usually, you don't need to do this."""

    @abstractmethod
    def on_close(
        self, data: CloseData[_IndexType], row: Series[Any]
    ) -> Iterable[LimitOrder | MarketOrder]:
        """Override this method to implement your strategy.
        Unfilled orders would be automatically canceled.

        Parameters
        ----------
        data: CloseData
            CloseData object.
        row: Series
            Row of the DataFrame passed to __call__."""
        yield from ()  # pragma: no cover
