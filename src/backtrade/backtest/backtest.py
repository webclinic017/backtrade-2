from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from functools import reduce
from logging import getLogger
from typing import Any, Generic, Iterable, final

import attrs
import joblib
import numpy as np
from exceptiongroup import ExceptionGroup
from pandas import DataFrame, Series
from tqdm import tqdm

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

    def _parrarel(
        self,
        df: DataFrame,
        *,
        maker_fee: float,
        taker_fee: float,
        balance_init: float = 1,
        name: str | None = None,
        n_splits: int = -1,
        logarithmic: bool = True,
    ) -> BacktestResult[_IndexType]:
        if n_splits == 0:
            raise ValueError("n_splits must be not 0")
        if n_splits < -joblib.parallel.cpu_count():
            raise ValueError(
                "n_splits must be greater than "
                + f"-cpu_count={-joblib.parallel.cpu_count()}"
            )
        if n_splits < 0:
            n_splits = joblib.parallel.cpu_count() + n_splits + 1

        df_split = np.array_split(df, n_splits)
        maker_fee_split = np.full(n_splits, maker_fee)
        taker_fee_split = np.full(n_splits, taker_fee)
        balance_init_split = np.full(n_splits, balance_init)
        name_split = np.full(n_splits, name)
        logarithmic_split = np.full(n_splits, logarithmic)
        results: list[BacktestResult[_IndexType]] | None = joblib.Parallel(
            n_jobs=-1, verbose=10
        )(
            joblib.delayed(self)(
                df,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                balance_init=balance_init,
                n_splits=1,
                logarithmic=logarithmic,
                name=name,
                use_tqdm=False,
            )
            for df, maker_fee, taker_fee, balance_init, logarithmic, name in zip(
                df_split,
                maker_fee_split,
                taker_fee_split,
                balance_init_split,
                logarithmic_split,
                name_split,
            )
        )
        if results is None:
            raise ValueError("Joblib returned None.")

        if logarithmic:
            result = reduce(
                lambda x, y: x + y * (x.equity_quote[-1] / y.equity_quote[0]), results
            )
            result = attrs.evolve(result, name=name)
        else:
            result = reduce(lambda x, y: x + y, results)
            result = attrs.evolve(
                result,
                name=name,
                equity_quote=result.equity_quote - balance_init * (n_splits - 1),
                balance_quote=result.balance_quote - balance_init * (n_splits - 1),
            )
        return result

    @final
    def __call__(
        self,
        df: DataFrame,
        *,
        maker_fee: float,
        taker_fee: float,
        balance_init: float = 1,
        name: str | None = None,
        n_splits: int = 1,
        logarithmic: bool = True,
        use_tqdm: bool = True,
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
        name : str, optional
            Name of the backtest, by default None
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

        # Multiprocessing
        if n_splits != 1:
            return self._parrarel(
                df,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                balance_init=balance_init,
                name=name,
                n_splits=n_splits,
                logarithmic=logarithmic,
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

        pbar = tqdm(df.iterrows(), total=len(df), disable=not use_tqdm)
        for index, row in pbar:
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

        finished_orders_exploded = finished_orders_history.explode().dropna()

        return BacktestResult(
            name=name,
            close=df["close"],
            position=position_history.rename("position"),
            position_quote=position_quote_history.rename("position_quote"),
            balance_quote=balance_quote_history.rename("balance_quote"),
            equity_quote=equity_quote_history.rename("equity_quote"),
            finished_orders=finished_orders_exploded.rename("finished_orders"),
            maker_fee_rate=maker_fee,
            taker_fee_rate=taker_fee,
            logarithmic=logarithmic,
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
