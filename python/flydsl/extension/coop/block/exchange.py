# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Portable, layout-directed redistribution of register-held block tiles."""

from ....expr.gpu import barrier, num_warp_threads
from ....expr.numeric import Numeric
from ....expr.primitive import get_scalar, make_layout
from ....expr.struct import Struct
from ....expr.typing import Array, Vector
from .. import warp as _dispatched_warp
from .._common import linear_thread_id, require_power_of_two
from ._spec import _block_shape

__all__ = ["BlockExchange"]

_CACHE = {}


def _layout_spec(name, threads, items, width):
    """Static Fly shape and stride, with a nested (lane, warp) thread mode."""
    if name == "blocked":
        return (threads, items), (items, 1)
    if name == "striped":
        return (threads, items), (1, threads)
    if name == "warp_striped":
        return ((width, threads // width), items), ((1, width * items), width)
    raise ValueError(f"unsupported BlockExchange arrangement: {name!r}")


class _BlockExchangeMeta(type):
    def __getitem__(cls, params):
        if cls.block_threads is not None:
            raise TypeError(f"{cls.__name__} is already specialized")
        if not isinstance(params, tuple) or len(params) != 3:
            raise TypeError("BlockExchange[dtype, block_size, items_per_thread]")
        dtype, block_size, items = params
        if (
            not isinstance(dtype, type)
            or not issubclass(dtype, Numeric)
            or getattr(dtype, "width", None) not in (8, 16, 32, 64)
        ):
            raise TypeError("BlockExchange dtype must be an 8-, 16-, 32-, or 64-bit Numeric type")
        block_size = _block_shape(block_size)
        threads = block_size[0] * block_size[1] * block_size[2]
        require_power_of_two(threads, "block thread count")
        require_power_of_two(items, "items_per_thread")
        if any(isinstance(dim, bool) for dim in block_size) or isinstance(items, bool):
            raise TypeError("block dimensions and items_per_thread must be integers, not bool")
        if threads > 1024:
            raise ValueError("BlockExchange supports at most 1024 threads")
        width = min(num_warp_threads(), threads)
        key = (cls, dtype, block_size, items, width)
        if key not in _CACHE:
            # Shift each row of 32 bank-sized groups by one group. Narrow
            # values share a four-byte group; wide values occupy one slot.
            # This is a bijection regardless of the target's bank geometry;
            # it does not promise conflict-free accesses on every target.
            # FIXME: in some cases bank size is 64 bytes
            bank_items = max(1, 32 // dtype.width)
            period = 32 * bank_items
            total = threads * items
            storage_items = total + ((total - 1) // period) * bank_items
            _CACHE[key] = type(
                f"{cls.__name__}[{dtype.__name__}, {block_size}, {items}]",
                (cls,),
                dict(
                    dtype=dtype,
                    block_size=block_size,
                    block_threads=threads,
                    items_per_thread=items,
                    warp_threads=width,
                    storage_items=storage_items,
                    _bank_items=bank_items,
                    _padding_period=period,
                    SharedStorage=Struct["buffer" : Array[dtype, storage_items, 16]],
                ),
            )
        return _CACHE[key]

    def __call__(cls, *args, **kwargs):
        raise TypeError("call a BlockExchange redistribution method on a specialization")


class BlockExchange(metaclass=_BlockExchangeMeta):
    """Redistribute a register tile using lane moves or padded shared memory.

    Usage::

        exchange = fx.coop.BlockExchange[fx.Float32, 256, 4]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        striped = exchange.blocked_to_striped(values, storage=storage)

    ``block_size`` is an integer or an ``(x, y, z)`` launch shape. Its product
    and ``items_per_thread`` must be powers of two. Threads are ordered with x
    varying fastest. ``values`` must be a flat Vector of the specialized dtype
    and length; the result has the same type and length.

    Blocked thread t holds logical items ``t * items_per_thread + i``;
    striped thread t holds ``t + i * block_threads``. Warp-striped applies
    striping independently to each contiguous logical warp. A block smaller
    than a hardware wave uses its block size as the logical warp width.

    All threads must reach the call together, with the specialized launch
    shape. Identity conversions stay in registers. Wave-local conversions use
    packed lane moves and need no storage. Cross-wave conversions require
    SharedStorage and one block barrier between stores and loads; insert a
    barrier before reusing that storage, including for an inverse exchange.
    Padding reduces bank conflicts for common arrangements but is not a
    target-independent guarantee of conflict-free accesses.
    """

    dtype = None
    block_size = None
    block_threads = None
    items_per_thread = None
    warp_threads = None
    storage_items = None
    SharedStorage = None
    warp_ops = _dispatched_warp

    @classmethod
    def blocked_to_striped(cls, values, *, storage=None):
        """Convert blocked ownership to striping across the block."""
        return cls._exchange(values, "blocked", "striped", storage)

    @classmethod
    def striped_to_blocked(cls, values, *, storage=None):
        """Convert block-striped ownership back to blocked ownership."""
        return cls._exchange(values, "striped", "blocked", storage)

    @classmethod
    def blocked_to_warp_striped(cls, values, *, storage=None):
        """Stripe each logical warp's contiguous tile independently."""
        return cls._exchange(values, "blocked", "warp_striped", storage)

    @classmethod
    def _storage_index(cls, rank):
        return rank + rank // cls._padding_period * cls._bank_items

    @classmethod
    def _exchange(cls, values, source, destination, storage):
        if cls.block_threads is None:
            raise TypeError("specialize first, e.g. BlockExchange[fx.Float32, 256, 4]")
        if not isinstance(values, Vector):
            raise TypeError("BlockExchange values must be a Vector")
        if values.dtype is not cls.dtype:
            raise TypeError(f"BlockExchange expects {cls.dtype.__name__} values")
        if values.shape != (cls.items_per_thread,):
            raise ValueError(f"BlockExchange expects a flat Vector of {cls.items_per_thread} items")
        # The named layouts are bijective. Their widest ownership movement
        # determines the lowering without enumerating the block's elements.
        if cls.items_per_thread == 1 or cls.block_threads == 1:
            return values
        if cls.block_threads == cls.warp_threads or "warp_striped" in (source, destination):
            return cls._exchange_warp(values, source, destination)
        if storage is None:
            raise TypeError("cross-wave BlockExchange requires SharedStorage")
        tid = linear_thread_id(cls.block_size)
        src = make_layout(*_layout_spec(source, cls.block_threads, cls.items_per_thread, cls.warp_threads))
        dst = make_layout(*_layout_spec(destination, cls.block_threads, cls.items_per_thread, cls.warp_threads))
        # A scalar thread coordinate is decomposed by Fly into (lane, warp)
        # for the nested warp-striped layout.
        for i in range(cls.items_per_thread):
            storage.buffer[cls._storage_index(get_scalar(src(tid, i)))] = values[i]
        barrier()
        return Vector.from_elements(
            [storage.buffer[cls._storage_index(get_scalar(dst(tid, i)))] for i in range(cls.items_per_thread)],
            cls.dtype,
        )

    @classmethod
    def warp_striped_to_blocked(cls, values, *, storage=None):
        """Undo warp striping independently inside each logical warp."""
        return cls._exchange(values, "warp_striped", "blocked", storage)

    @classmethod
    def _exchange_warp(cls, values, source, destination):
        lane = linear_thread_id(cls.block_size) % cls.warp_threads
        outputs = []
        for item in range(cls.items_per_thread):
            # Within a wave, both striped arrangements have the same ranks.
            rank = lane * cls.items_per_thread + item if destination == "blocked" else lane + item * cls.warp_threads
            if source == "blocked":
                source_lane, source_item = rank // cls.items_per_thread, rank % cls.items_per_thread
            else:
                source_lane, source_item = rank % cls.warp_threads, rank // cls.warp_threads
            peer = cls.warp_ops.warp_permute(values, source_lane, width=cls.warp_threads)
            outputs.append(peer[source_item])
        return Vector.from_elements(outputs, cls.dtype)
