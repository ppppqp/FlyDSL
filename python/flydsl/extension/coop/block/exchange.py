# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide redistribution of register-held items."""

from ....expr.gpu import barrier, num_warp_threads
from ....expr.numeric import Numeric
from ....expr.primitive import get_scalar, make_layout
from ....expr.struct import Struct
from ....expr.typing import Array, Vector
from .._common import linear_thread_id, require_power_of_two
from ._spec import _block_shape

__all__ = ["BlockExchange"]


_CACHE = {}
# This padding model is only valid for AMD targets with 32 four-byte LDS
# banks (for example, gfx942). Some newer targets have 64 banks; until bank
# geometry is target metadata, this implementation must not be assumed to
# provide conflict-avoiding padding on those targets.
_LDS_BANKS = 32
_LDS_BANK_BYTES = 4


class _BlockExchangeMeta(type):
    """Gives ``BlockExchange`` its ``[dtype, block_size, items_per_thread]`` syntax."""

    def __getitem__(cls, params):
        if cls.block_threads is not None:
            raise TypeError(f"{cls.__name__} is already specialized")
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) != 3:
            raise TypeError(f"{cls.__name__}[dtype, block_size, items_per_thread]")

        dtype, block_size, items_per_thread = params
        if not (isinstance(dtype, type) and issubclass(dtype, Numeric)):
            raise TypeError(f"dtype must be a Numeric subclass, got {dtype!r}")
        if dtype.width < 8 or dtype.width % 8:
            raise TypeError(f"dtype must be byte-addressable, got {dtype.__name__} ({dtype.width} bits)")

        block_size = _block_shape(block_size)
        block_threads = block_size[0] * block_size[1] * block_size[2]
        require_power_of_two(block_threads, "block_dim_x * block_dim_y * block_dim_z")
        require_power_of_two(items_per_thread, "items_per_thread")

        warp_threads = min(num_warp_threads(), block_threads)
        key = (cls, dtype, block_size, items_per_thread, warp_threads)
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        # rocPRIM's padding scheme shifts each 128-byte LDS row by one
        # 32-bit bank unit. For narrow elements that unit spans several items.
        element_bytes = dtype.width // 8
        bank_items = max(1, _LDS_BANK_BYTES // element_bytes)
        total_items = block_threads * items_per_thread
        padding_items = total_items // _LDS_BANKS if items_per_thread >= 2 else 0
        storage_items = total_items + padding_items

        specialized = type(
            f"{cls.__name__}[{dtype.__name__}, {block_threads}, {items_per_thread}]",
            (cls,),
            {
                "dtype": dtype,
                "block_size": block_size,
                "block_threads": block_threads,
                "items_per_thread": items_per_thread,
                "warp_threads": warp_threads,
                "num_warps": block_threads // warp_threads,
                "bank_items": bank_items,
                "padding_items": padding_items,
                "storage_items": storage_items,
                "SharedStorage": Struct["buffer" : Array[dtype, storage_items, 16]],
            },
        )
        _CACHE[key] = specialized
        return specialized

    def __call__(cls, *args, **kwargs):
        raise TypeError("BlockExchange is a namespace; call one of its redistribution methods")


class BlockExchange(metaclass=_BlockExchangeMeta):
    """Portable block-wide exchange through padded LDS.

    Specialize the exchange for the value type, launch shape, and number of
    register items owned by each thread::

        exchange = fx.coop.BlockExchange[fx.Float32, 256, 4]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        striped = exchange.blocked_to_striped(values, storage=storage)

    ``values`` must be a thread-local :class:`~flydsl.expr.typing.Vector` with
    exactly ``items_per_thread`` elements of ``dtype``. The result is a flat
    vector with the same type and length.

    Every block thread must reach an exchange call together. Each portable
    transformation performs one block barrier between staging its inputs and
    reading its outputs. Reusing the storage for another collective requires a
    barrier after the call because the final reads are not followed by one.

    Blocked, striped, and warp-striped arrangements are represented by static
    Fly layouts over ``(thread, item)`` or ``(warp, lane, item)`` coordinates.
    The initial implementation deliberately supports power-of-two block sizes
    and item counts only.
    """

    dtype = None
    block_size = None
    block_threads = None
    items_per_thread = None
    warp_threads = None
    num_warps = None
    bank_items = None
    padding_items = None
    storage_items = None
    SharedStorage = None

    @classmethod
    def blocked_layout(cls):
        return make_layout(
            (cls.block_threads, cls.items_per_thread),
            (cls.items_per_thread, 1),
        )

    @classmethod
    def striped_layout(cls):
        return make_layout(
            (cls.block_threads, cls.items_per_thread),
            (1, cls.block_threads),
        )

    @classmethod
    def warp_striped_layout(cls):
        return make_layout(
            (cls.num_warps, cls.warp_threads, cls.items_per_thread),
            (cls.warp_threads * cls.items_per_thread, 1, cls.warp_threads),
        )

    @classmethod
    def blocked_to_striped(cls, values, *, storage):
        """Redistribute a blocked arrangement across the whole block."""
        tid = linear_thread_id(cls.block_size)
        source = cls.blocked_layout()
        destination = cls.striped_layout()
        for item in range(cls.items_per_thread):
            storage.buffer[cls._storage_index(get_scalar(source(tid, item)))] = values[item]
        barrier()
        return Vector.from_elements(
            [
                storage.buffer[cls._storage_index(get_scalar(destination(tid, item)))]
                for item in range(cls.items_per_thread)
            ],
            cls.dtype,
        )

    @classmethod
    def striped_to_blocked(cls, values, *, storage):
        """Redistribute a block-striped arrangement back to blocked."""
        tid = linear_thread_id(cls.block_size)
        source = cls.striped_layout()
        destination = cls.blocked_layout()
        for item in range(cls.items_per_thread):
            storage.buffer[cls._storage_index(get_scalar(source(tid, item)))] = values[item]
        barrier()
        return Vector.from_elements(
            [
                storage.buffer[cls._storage_index(get_scalar(destination(tid, item)))]
                for item in range(cls.items_per_thread)
            ],
            cls.dtype,
        )

    @classmethod
    def blocked_to_warp_striped(cls, values, *, storage):
        """Redistribute blocked items independently inside each logical warp."""
        tid = linear_thread_id(cls.block_size)
        warp_id = tid // cls.warp_threads
        lane = tid % cls.warp_threads
        source = cls.blocked_layout()
        destination = cls.warp_striped_layout()
        for item in range(cls.items_per_thread):
            storage.buffer[cls._storage_index(get_scalar(source(tid, item)))] = values[item]
        barrier()
        return Vector.from_elements(
            [
                storage.buffer[cls._storage_index(get_scalar(destination(warp_id, lane, item)))]
                for item in range(cls.items_per_thread)
            ],
            cls.dtype,
        )

    @classmethod
    def warp_striped_to_blocked(cls, values, *, storage):
        """Redistribute warp-striped items back to block-wide blocked ownership."""
        tid = linear_thread_id(cls.block_size)
        warp_id = tid // cls.warp_threads
        lane = tid % cls.warp_threads
        source = cls.warp_striped_layout()
        destination = cls.blocked_layout()
        for item in range(cls.items_per_thread):
            storage.buffer[cls._storage_index(get_scalar(source(warp_id, lane, item)))] = values[item]
        barrier()
        return Vector.from_elements(
            [
                storage.buffer[cls._storage_index(get_scalar(destination(tid, item)))]
                for item in range(cls.items_per_thread)
            ],
            cls.dtype,
        )

    @classmethod
    def _storage_index(cls, logical_index):
        """
        Map a logical rank into the padded LDS buffer.
        TODO: swizzling as an alternative?
        """
        if cls.padding_items == 0:
            return logical_index
        period = _LDS_BANKS * cls.bank_items
        return logical_index + (logical_index // period) * cls.bank_items
