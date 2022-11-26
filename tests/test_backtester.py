from typing import Any, Iterable, Union
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame, Series

from backtrade import Backtester, CloseData, LimitOrder, MarketOrder, _IndexType


class TestBacktester(TestCase):
    def test_backtester(self):
        class MyBacktest(Backtester):
            def init(self):
                pass

            def on_close(
                self, close_data: "CloseData[_IndexType]", row: "Series[Any]"
            ) -> "Iterable[Union[MarketOrder, LimitOrder]]":
                yield MarketOrder(size=1 / close_data.close)
                yield MarketOrder(size=-1 / close_data.close)

        n = 1000
        df = DataFrame({"close": np.random.rand(n + 1)})
        df["open"] = df["close"].shift(1)
        df.dropna(inplace=True)
        assert df.shape[0] == n
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.rand(n) * 0.1)
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.rand(n) * 0.1)
        taker_fee = 0.001
        balance_init = 1
        res = MyBacktest()(df, maker_fee=taker_fee, taker_fee=taker_fee)
        res.plot()
        plt.show()
        self.assertTrue((res.position == 0).all())
        self.assertAlmostEqual(
            res.equity_quote.iat[-1], balance_init - taker_fee * 2 * (n - 1)
        )
