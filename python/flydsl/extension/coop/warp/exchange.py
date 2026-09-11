# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Portable warp-local register movement."""

from ....expr.gpu import lane_id, num_warp_threads, shuffle_idx
from ....expr.numeric import Numeric
from ....expr.typing import Vector
from .._common import resolve_warp_width

__all__ = ["warp_permute"]


def _absolute_source_lane(source_lane, width):
    """Turn a lane relative to its logical warp into a physical lane."""
    if width == num_warp_threads():
        return source_lane
    return (lane_id() // width) * width + source_lane


def warp_permute(value, source_lane, *, width=None):
    """Read the scalar or vector held by ``source_lane`` in this warp.

    ``source_lane`` is relative to the calling lane's logical ``width`` group.
    Each lane may request a different source. The portable implementation
    shuffles vector elements separately; target overrides may pack them.
    """
    width = resolve_warp_width(width, "warp_permute width")
    source_lane = _absolute_source_lane(source_lane, width)
    if isinstance(value, Vector):
        return Vector.from_elements([shuffle_idx(item, source_lane, width) for item in value], value.dtype)
    if isinstance(value, Numeric):
        return shuffle_idx(value, source_lane, width)
    raise TypeError(f"value must be a Numeric or Vector, got {type(value).__name__}")
