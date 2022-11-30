from unittest import TestCase

from backtrade.logic import (
    FinishedOrder,
    FinishedOrderState,
    ProcessBuyOrderArgs,
    ProcessOrderArgs,
    ProcessSellOrderArgs,
    ToLimitOrderArgs,
    process_buy_order,
    process_order,
    process_sell_order,
    to_limit_order,
)
from backtrade.order import LimitOrder, MarketOrder


class TestLogic(TestCase):
    def test_to_limit_order(self):
        limit_order = LimitOrder(size=1, price=1, post_only=False)
        self.assertEqual(
            to_limit_order(ToLimitOrderArgs(order=limit_order, last_close=1)),
            limit_order,
        )

        market_order = MarketOrder(size=1)
        self.assertEqual(
            to_limit_order(ToLimitOrderArgs(order=market_order, last_close=1)),
            LimitOrder(size=1, price=2, post_only=False),
        )
        market_order = MarketOrder(size=-1)
        self.assertEqual(
            to_limit_order(ToLimitOrderArgs(order=market_order, last_close=1)),
            LimitOrder(size=-1, price=0.5, post_only=False),
        )
        market_order = MarketOrder(size=0)
        self.assertEqual(
            to_limit_order(ToLimitOrderArgs(order=market_order, last_close=1)),
            LimitOrder(size=0, price=1, post_only=False),
        )

    def test_process_buy_order(self):
        order = LimitOrder(size=1, price=1, post_only=False)
        self.assertEqual(
            process_buy_order(
                ProcessBuyOrderArgs(
                    order=order,
                    last_close=1.01,
                    low=0.5,
                    taker_fee=0,
                    maker_fee=0,
                    index=0,
                )
            ),
            FinishedOrder(
                index=0,
                order=order,
                balance_decrement=1,
                fee=0,
                state=FinishedOrderState.FilledMaker,
            ),
        )
        self.assertEqual(
            process_buy_order(
                ProcessBuyOrderArgs(
                    order=order,
                    last_close=1,
                    low=0.5,
                    taker_fee=0,
                    maker_fee=0,
                    index=0,
                )
            ),
            FinishedOrder(
                index=0,
                order=order,
                balance_decrement=1,
                fee=0,
                state=FinishedOrderState.FilledTaker,
            ),
        )

    def test_process_sell_order(self):
        order = LimitOrder(size=-1, price=1, post_only=False)
        self.assertEqual(
            process_sell_order(
                ProcessSellOrderArgs(
                    order=order,
                    last_close=0.99,
                    high=2,
                    taker_fee=0,
                    maker_fee=0,
                    index=0,
                )
            ),
            FinishedOrder(
                index=0,
                order=order,
                balance_decrement=-1,
                fee=0,
                state=FinishedOrderState.FilledMaker,
            ),
        )
        self.assertEqual(
            process_sell_order(
                ProcessSellOrderArgs(
                    order=order, last_close=1, high=2, taker_fee=0, maker_fee=0, index=0
                )
            ),
            FinishedOrder(
                index=0,
                order=order,
                balance_decrement=-1,
                fee=0,
                state=FinishedOrderState.FilledTaker,
            ),
        )

    def test_errors(self):
        def create_args(size: int) -> "ProcessOrderArgs[int, MarketOrder]":
            return ProcessOrderArgs(
                order=MarketOrder(size=size),
                last_close=1,
                high=2,
                low=0.5,
                taker_fee=0,
                maker_fee=0,
                index=0,
            )

        with self.assertRaises(ValueError):
            process_buy_order(create_args(-1))
        with self.assertRaises(ValueError):
            process_sell_order(create_args(1))
        with self.assertRaises(ValueError):
            process_order(create_args(0))
