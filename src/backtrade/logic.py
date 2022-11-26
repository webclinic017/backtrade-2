from enum import Enum
from typing import Generic, Union

import attrs

from .dtypes import LimitOrder, MarketOrder, _IndexType, _OrderType


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
    state: FinishedOrderState

    @property
    def filled(self) -> bool:
        return self.state in (
            FinishedOrderState.FilledTaker,
            FinishedOrderState.FilledMaker,
        )


def to_limit_order(
    order: Union[LimitOrder, MarketOrder],
    base_price: float,
) -> LimitOrder:
    if isinstance(order, LimitOrder):
        return order
    if order.size > 0:
        price = base_price * 2
    elif order.size < 0:
        price = base_price / 2
    else:
        price = base_price
    order = LimitOrder(
        size=order.size,
        price=price,
        post_only=False,
    )
    return order


def process_buy_order(
    order: _OrderType,
    open: float,
    low: float,
    taker_fee: float,
    maker_fee: float,
    index: _IndexType,
) -> "FinishedOrder[_IndexType, _OrderType]":
    if order.size <= 0:
        raise ValueError("Buy order size must be positive")
    limit_order = to_limit_order(order, open)
    if limit_order.price >= open:
        # taker
        if limit_order.post_only:
            return FinishedOrder(
                index=index,
                order=order,
                balance_decrement=0,
                state=FinishedOrderState.CancelledPostOnly,
            )
        else:
            return FinishedOrder(
                index=index,
                order=order,
                balance_decrement=order.size * open * (1 + taker_fee),
                state=FinishedOrderState.FilledTaker,
            )
    elif limit_order.price >= low:
        return FinishedOrder(
            index=index,
            order=order,
            balance_decrement=order.size * limit_order.price * (1 + maker_fee),
            state=FinishedOrderState.FilledMaker,
        )
    else:
        return FinishedOrder(
            index=index,
            order=order,
            balance_decrement=0,
            state=FinishedOrderState.CancelledNotFilled,
        )


def process_sell_order(
    order: _OrderType,
    open: float,
    high: float,
    taker_fee: float,
    maker_fee: float,
    index: _IndexType,
) -> "FinishedOrder[_IndexType, _OrderType]":
    if order.size >= 0:
        raise ValueError("Sell order size must be negative")
    limit_order = to_limit_order(order, open)
    if limit_order.price <= open:
        # taker
        if limit_order.post_only:
            return FinishedOrder(
                index=index,
                order=order,
                balance_decrement=0,
                state=FinishedOrderState.CancelledPostOnly,
            )
        else:
            return FinishedOrder(
                index=index,
                order=order,
                balance_decrement=order.size * open * (1 - taker_fee),
                state=FinishedOrderState.FilledTaker,
            )
    elif limit_order.price <= high:
        return FinishedOrder(
            index=index,
            order=order,
            balance_decrement=order.size * limit_order.price * (1 - maker_fee),
            state=FinishedOrderState.FilledMaker,
        )
    else:
        return FinishedOrder(
            index=index,
            order=order,
            balance_decrement=0,
            state=FinishedOrderState.CancelledNotFilled,
        )


def process_order(
    order: _OrderType,
    open: float,
    high: float,
    low: float,
    taker_fee: float,
    maker_fee: float,
    index: _IndexType,
) -> "FinishedOrder[_IndexType, _OrderType]":
    if order.size > 0:
        return process_buy_order(order, open, low, taker_fee, maker_fee, index)
    elif order.size < 0:
        return process_sell_order(order, open, high, taker_fee, maker_fee, index)
    else:
        raise ValueError("Order size must be non-zero", order)
