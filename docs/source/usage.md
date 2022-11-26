# Usage

```python
from backtrade import Backtester, CloseData, LimitOrder, MarketOrder, _IndexType

# Define my strategy
class MyBacktest(Backtester):
    def init(self):
        pass

    def next(
        self, close_data: "CloseData[_IndexType]", row: "Series[Any]"
    ) -> "Iterable[Union[MarketOrder, LimitOrder]]":
        yield MarketOrder(size=1 / close_data.close)
        yield MarketOrder(size=-1 / close_data.close)

# Run backtest and plot results
bt = MyBacktest()
bt(df, maker_fee=-0.025 * 0.01, taker_fee=0.001).plot()
```

For more examples, see the [example notebook](https://github.com/34j/backtrade/blob/master/example/example.ipynb).
