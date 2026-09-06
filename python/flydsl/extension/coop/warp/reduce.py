# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-wide reduction — the portable form."""

from ....expr.gpu import shuffle_xor
from .._common import combine, resolve_warp_width

__all__ = ["warp_reduce"]


def warp_reduce(value, op, *, width=None):
    """Reduce *value* across *width* lanes; every participating lane gets the result.

    *width* defaults to the target's warp size; a narrower power-of-two width
    reduces independent groups of lanes side by side.

    *op* must be commutative, as well as associative. A butterfly folds each
    lane's own partial on the left and its XOR partner's on the right, and at
    step ``k`` a lane's own half is the lower one only where its bit ``k`` is
    clear. Lane 0 therefore folds the group in lane order at every width, and
    every other lane folds some permutation of it, so the lanes agree on one
    result only if reordering does not change it.

    Every :class:`~flydsl.expr.typing.ReductionOp` this library accepts today
    is commutative, so nothing currently depends on the distinction — see
    :func:`~flydsl.extension.coop._common.combine`, which is where a
    non-commutative *op* would first become expressible.
    """

    """
    NOTE:
    For a 64-lane wave, tracing expands this into six shuffle-and-combine stages at offests:
    1, 2, 4, 8, 16, 32
    There is no runtime while loop in the resulting GPU.
    Butterfly reduction. For 8 lanes:
    Initial:   [a0 a1 a2 a3 a4 a5 a6 a7]
    XOR 1 (flips bit 0):     pairs 0↔1, 2↔3, 4↔5, 6↔7
    XOR 2 (flips bit 1):     pairs 0↔2, 1↔3, 4↔6, 5↔7
    XOR 4 (flips bit 2):     pairs 0↔4, 1↔5, 2↔6, 3↔7
    """
    width = resolve_warp_width(width, "warp_reduce width")
    offset = 1
    while offset < width:
        value = combine(op, value, shuffle_xor(value, offset, width))
        offset <<= 1
    return value
