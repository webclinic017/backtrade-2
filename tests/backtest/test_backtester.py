from typing import Any, Iterable, Union
from unittest import TestCase

import numpy as np
from exceptiongroup import ExceptionGroup
from pandas import DataFrame, Series

from backtrade import Backtester, CloseData, LimitOrder, MarketOrder, _IndexType
from backtrade.logic import FinishedOrderState


def generate_random_ohlcv(n: int) -> DataFrame:
    df = DataFrame({"close": np.random.rand(n + 1)})
    df["open"] = df["close"].shift(1)
    df.dropna(inplace=True)
    assert len(df) == n
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.rand(n) * 0.1)
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.rand(n) * 0.1)
    return df


class TestBacktester(TestCase):
    def setUp(self) -> None:
        self.n = np.random.randint(50, 150)
        self.maker_fee = np.random.rand() * 0.01 - 0.005
        self.taker_fee = np.random.rand() * 0.01
        self.balance_init = np.random.rand() * 1000

    def test_market_both(self):
        class MyBacktest(Backtester[_IndexType]):
            def init(self):
                pass

            def on_close(
                self, close_data: "CloseData[_IndexType]", row: "Series[Any]"
            ) -> "Iterable[Union[MarketOrder, LimitOrder]]":
                yield MarketOrder(size=1 / close_data.close)
                yield MarketOrder(size=-1 / close_data.close)

        df = generate_random_ohlcv(self.n)
        bt: "Backtester[int]" = MyBacktest()
        res = bt(
            df,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            balance_init=self.balance_init,
        )
        res.plot()
        istaker = (
            res.flatten_finished_orders.apply(lambda x: x.state)
            == FinishedOrderState.FilledTaker
        )
        self.assertEqual(len(istaker[~istaker]), 0)
        self.assertTrue((res.position == 0).all())
        self.assertAlmostEqual(
            res.equity_quote.iat[-1],
            self.balance_init - self.taker_fee * 2 * (self.n - 1),
        )

    def test_limit_both(self):
        class MyBacktest(Backtester[_IndexType]):
            def init(self):
                pass

            def on_close(
                self, close_data: "CloseData[_IndexType]", row: "Series[Any]"
            ) -> "Iterable[Union[MarketOrder, LimitOrder]]":
                yield LimitOrder(size=1, price=0, post_only=False)
                yield LimitOrder(size=-1, price=99999, post_only=False)

        df = generate_random_ohlcv(self.n)
        bt: "Backtester[int]" = MyBacktest()
        res = bt(
            df,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            balance_init=self.balance_init,
        )
        res.plot()
        self.assertTrue((res.position == 0).all())
        self.assertTrue((res.equity_quote == self.balance_init).all())

    def test_postonly(self):
        class MyBacktest(Backtester[_IndexType]):
            def init(self):
                pass

            def on_close(
                self, close_data: "CloseData[_IndexType]", row: "Series[Any]"
            ) -> "Iterable[Union[MarketOrder, LimitOrder]]":
                yield LimitOrder(size=1, price=99999, post_only=True)
                yield LimitOrder(size=-1, price=0, post_only=True)

        df = generate_random_ohlcv(self.n)
        bt: "Backtester[int]" = MyBacktest()
        res = bt(
            df,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            balance_init=self.balance_init,
        )
        res.plot()
        self.assertTrue((res.position == 0).all())
        self.assertTrue((res.equity_quote == self.balance_init).all())

    def test_zero_orders(self):
        class MyBacktest(Backtester[_IndexType]):
            def init(self):
                pass

            def on_close(
                self, close_data: "CloseData[_IndexType]", row: "Series[Any]"
            ) -> "Iterable[Union[MarketOrder, LimitOrder]]":
                yield LimitOrder(size=0, price=99999, post_only=True)
                yield MarketOrder(size=0)

        df = generate_random_ohlcv(self.n)
        bt: "Backtester[int]" = MyBacktest()
        res = bt(
            df,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            balance_init=self.balance_init,
        )
        res.plot()
        self.assertTrue((res.position == 0).all())
        self.assertTrue((res.equity_quote == self.balance_init).all())

    def test_errors(self):
        class EmptyBacktest(Backtester[_IndexType]):
            def on_close(
                self, close_data: "CloseData[_IndexType]", row: "Series[Any]"
            ) -> "Iterable[Union[MarketOrder, LimitOrder]]":
                yield from ()

        df = DataFrame(
            {
                "Close": np.random.rand(self.n) - 1,
                "Open": np.random.rand(self.n) - 1,
                "Low": np.random.rand(self.n) - 1,
                "High": np.random.rand(self.n) - 1,
            }
        )
        df.index = np.random.randint(0, 10, self.n)
        bt: "Backtester[int]" = EmptyBacktest()
        with self.assertRaises(ExceptionGroup) as ecm:
            with self.assertWarns(UserWarning):
                bt(df, maker_fee=0, taker_fee=-1, balance_init=-1)
        self.assertEqual(len(ecm.exception.exceptions), 12)

        df = DataFrame()
        with self.assertRaises(ExceptionGroup) as ecm:
            bt(df, maker_fee=0, taker_fee=0, balance_init=1)
        self.assertEqual(len(ecm.exception.exceptions), 1)
