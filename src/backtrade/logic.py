from typing import Generic

import attrs

from .finished_order import FinishedOrder, FinishedOrderState
from .order import LimitOrder, _IndexType, _OrderType


@attrs.frozen(kw_only=True, slots=False)
class ToLimitOrderArgs(Generic[_OrderType]):
    order: _OrderType
    last_close: float


@attrs.frozen(kw_only=True, slots=False)
class ProcessOrderArgsBase(
    ToLimitOrderArgs[_OrderType], Generic[_IndexType, _OrderType]
):
    taker_fee: float
    maker_fee: float
    index: _IndexType


@attrs.frozen(kw_only=True, slots=False)
class ProcessSellOrderArgs(ProcessOrderArgsBase[_IndexType, _OrderType]):
    high: float


@attrs.frozen(kw_only=True, slots=False)
class ProcessBuyOrderArgs(ProcessOrderArgsBase[_IndexType, _OrderType]):
    low: float


def to_limit_order(args: ToLimitOrderArgs[_OrderType]) -> LimitOrder:
    if isinstance(args.order, LimitOrder):
        return args.order
    if args.order.size > 0:
        price = args.last_close * 2
    elif args.order.size < 0:
        price = args.last_close / 2
    else:
        price = args.last_close
    return LimitOrder(
        size=args.order.size,
        price=price,
        post_only=False,
    )


def process_buy_order(
    args: ProcessBuyOrderArgs[_IndexType, _OrderType]
) -> "FinishedOrder[_IndexType, _OrderType]":
    if args.order.size <= 0:
        raise ValueError("Buy order size must be positive")
    limit_order = to_limit_order(args)
    if limit_order.price >= args.last_close:
        # taker
        if limit_order.post_only:
            return FinishedOrder(
                index=args.index,  # type: ignore
                order=args.order,  # type: ignore
                balance_decrement=0,
                fee=0,
                state=FinishedOrderState.CancelledPostOnly,
            )
        else:
            return FinishedOrder(
                index=args.index,  # type: ignore
                order=args.order,  # type: ignore
                balance_decrement=args.order.size
                * args.last_close
                * (1 + args.taker_fee),
                fee=args.order.size * args.last_close * args.taker_fee,
                state=FinishedOrderState.FilledTaker,
            )
    elif limit_order.price >= args.low:
        return FinishedOrder(
            index=args.index,  # type: ignore
            order=args.order,  # type: ignore
            balance_decrement=args.order.size
            * limit_order.price
            * (1 + args.maker_fee),
            fee=args.order.size * limit_order.price * args.maker_fee,
            state=FinishedOrderState.FilledMaker,
        )
    else:
        return FinishedOrder(
            index=args.index,  # type: ignore
            order=args.order,  # type: ignore
            balance_decrement=0,
            fee=0,
            state=FinishedOrderState.CancelledNotFilled,
        )


def process_sell_order(
    args: ProcessSellOrderArgs[_IndexType, _OrderType]
) -> "FinishedOrder[_IndexType, _OrderType]":
    if args.order.size >= 0:
        raise ValueError("Sell order size must be negative")
    limit_order = to_limit_order(args)
    if limit_order.price <= args.last_close:
        # taker
        if limit_order.post_only:
            return FinishedOrder(
                index=args.index,  # type: ignore
                order=args.order,  # type: ignore
                balance_decrement=0,
                fee=0,
                state=FinishedOrderState.CancelledPostOnly,
            )
        else:
            return FinishedOrder(
                index=args.index,  # type: ignore
                order=args.order,  # type: ignore
                balance_decrement=args.order.size
                * args.last_close
                * (1 - args.taker_fee),
                fee=args.order.size * args.last_close * args.taker_fee,
                state=FinishedOrderState.FilledTaker,
            )
    elif limit_order.price <= args.high:
        return FinishedOrder(
            index=args.index,  # type: ignore
            order=args.order,  # type: ignore
            balance_decrement=args.order.size
            * limit_order.price
            * (1 - args.maker_fee),
            fee=args.order.size * limit_order.price * args.maker_fee,
            state=FinishedOrderState.FilledMaker,
        )
    else:
        return FinishedOrder(
            index=args.index,  # type: ignore
            order=args.order,  # type: ignore
            balance_decrement=0,
            fee=0,
            state=FinishedOrderState.CancelledNotFilled,
        )


@attrs.frozen(kw_only=True, slots=False)
class ProcessOrderArgs(
    ProcessBuyOrderArgs[_IndexType, _OrderType],
    ProcessSellOrderArgs[_IndexType, _OrderType],
):
    pass


def process_order(
    args: ProcessOrderArgs[_IndexType, _OrderType]
) -> "FinishedOrder[_IndexType, _OrderType]":
    if args.order.size > 0:
        return process_buy_order(args)
    elif args.order.size < 0:
        return process_sell_order(args)
    else:
        raise ValueError("Order size must be non-zero", args.order)
