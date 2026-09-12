# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Bit-preserving lane permutation of scalars and register vectors."""

from ....expr.gpu import lane_id, num_warp_threads, shuffle_idx, shuffle_xor
from ....expr.numeric import Int32, Numeric
from ....expr.typing import Vector
from .._common import resolve_warp_width

__all__ = ["warp_permute", "warp_permute_xor"]


def _permute(value, source_lane, width, move_word, *, xor=False):
    width = resolve_warp_width(width, "warp_permute width")
    if not isinstance(value, (Numeric, Vector)) or value.dtype.width not in (8, 16, 32, 64):
        raise TypeError("warp_permute expects an 8-, 16-, 32-, or 64-bit scalar or Vector")
    if isinstance(source_lane, int) and not 0 <= source_lane < width:
        raise ValueError(f"source_lane must be in [0, {width})")
    if xor and not isinstance(source_lane, int):
        raise TypeError("warp_permute_xor requires a static integer lane mask")
    if width == 1 or (xor and source_lane == 0):
        return value
    dtype = value.dtype
    values = value.reshape((value.numel,)) if isinstance(value, Vector) else Vector.from_elements([value], dtype)
    count = values.numel
    # Pack narrow elements together, padding only the last register. Wide
    # elements split into two words. No numeric conversion crosses the wave.
    per_word = max(1, 32 // dtype.width)
    padding = (-count) % per_word
    if padding:
        values = Vector.from_elements([*values, *[dtype(0) for _ in range(padding)]], dtype)
    words = values.bitcast(Int32)
    # A full-wave source is already an absolute lane. Avoid materializing
    # lane_id and a redundant group-base calculation in this common path.
    source = source_lane if xor or width == num_warp_threads() else (lane_id() // width) * width + source_lane
    moved = Vector.from_elements([move_word(word, source) for word in words], Int32).bitcast(dtype)
    if padding:
        moved = moved.shuffle(moved, list(range(count)))
    return moved.reshape(value.shape) if isinstance(value, Vector) else moved[0]


def _shuffle_word(word, source):
    return Int32(shuffle_idx(word, source, num_warp_threads()))


def warp_permute(value, source_lane, *, width=None):
    """Read a scalar or whole Vector from a lane in the same logical warp.

    ``source_lane`` is relative to each aligned ``width``-lane group and must
    be in ``[0, width)``. Width defaults to the hardware wave; narrower widths
    must be powers of two. All source lanes must be active and execute this
    call with the receiving lanes. Partial physical waves are supported when
    every requested source lane is active. The result preserves shape, dtype,
    and bits, including for narrow and 64-bit elements. No LDS or block
    barrier is used.
    """
    return _permute(value, source_lane, width, _shuffle_word)


def _shuffle_xor_word(word, mask):
    return Int32(shuffle_xor(word, mask, num_warp_threads()))


def warp_permute_xor(value, lane_mask, *, width=None):
    """Read a scalar or Vector from lane ``lane_id ^ lane_mask``.

    The static integer mask must be in ``[0, width)``. The active-lane,
    dtype, shape, and synchronization contract is the same as warp_permute.
    A constant mask permits target-specific moves without a lane address.
    """
    return _permute(value, lane_mask, width, _shuffle_xor_word, xor=True)
