# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide redistribution of register-held items."""

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
# This padding model is only valid for AMD targets with 32 four-byte LDS
# banks (for example, gfx942). Some newer targets have 64 banks; until bank
# geometry is target metadata, this implementation must not be assumed to
# provide conflict-avoiding padding on those targets.
_LDS_BANKS = 32
_LDS_BANK_BYTES = 4

_BLOCKED = "blocked"
_STRIPED = "striped"
_WARP_STRIPED = "warp_striped"
_REGISTER = "register"
_WARP = "warp"
_LDS = "lds"


def _rank(layout, thread, item, block_threads, items_per_thread, warp_threads):
    if layout == _BLOCKED:
        return thread * items_per_thread + item
    if layout == _STRIPED:
        return thread + item * block_threads
    warp_id, lane = divmod(thread, warp_threads)
    return warp_id * warp_threads * items_per_thread + lane + item * warp_threads


def _owner(layout, rank, block_threads, items_per_thread, warp_threads):
    if layout == _BLOCKED:
        return divmod(rank, items_per_thread)
    if layout == _STRIPED:
        item, thread = divmod(rank, block_threads)
        return thread, item
    warp_items = warp_threads * items_per_thread
    warp_id, warp_rank = divmod(rank, warp_items)
    item, lane = divmod(warp_rank, warp_threads)
    return warp_id * warp_threads + lane, item


def _plan_exchange(source, destination, block_threads, items_per_thread, warp_threads):
    """Classify a static thread/value permutation by its widest movement."""
    register_local = True
    warp_local = True
    for thread in range(block_threads):
        for item in range(items_per_thread):
            rank = _rank(destination, thread, item, block_threads, items_per_thread, warp_threads)
            source_thread, source_item = _owner(source, rank, block_threads, items_per_thread, warp_threads)
            register_local &= source_thread == thread
            warp_local &= source_thread // warp_threads == thread // warp_threads
            if source_item < 0 or source_item >= items_per_thread:
                raise ValueError(f"non-bijective {source} -> {destination} exchange")
    if register_local:
        return _REGISTER
    if warp_local:
        return _WARP
    return _LDS


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
        blocked_to_striped_lowering = _plan_exchange(_BLOCKED, _STRIPED, block_threads, items_per_thread, warp_threads)
        striped_to_blocked_lowering = _plan_exchange(_STRIPED, _BLOCKED, block_threads, items_per_thread, warp_threads)
        blocked_to_warp_striped_lowering = _plan_exchange(
            _BLOCKED, _WARP_STRIPED, block_threads, items_per_thread, warp_threads
        )
        warp_striped_to_blocked_lowering = _plan_exchange(
            _WARP_STRIPED, _BLOCKED, block_threads, items_per_thread, warp_threads
        )

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
                "_blocked_to_striped_lowering": blocked_to_striped_lowering,
                "_striped_to_blocked_lowering": striped_to_blocked_lowering,
                "_blocked_to_warp_striped_lowering": blocked_to_warp_striped_lowering,
                "_warp_striped_to_blocked_lowering": warp_striped_to_blocked_lowering,
                "SharedStorage": Struct["buffer" : Array[dtype, storage_items, 16]],
            },
        )
        _CACHE[key] = specialized
        return specialized

    def __call__(cls, *args, **kwargs):
        raise TypeError("BlockExchange is a namespace; call one of its redistribution methods")


class BlockExchange(metaclass=_BlockExchangeMeta):
    """Layout-directed block exchange through registers, wave moves, or LDS.

    Specialize the exchange for the value type, launch shape, and number of
    register items owned by each thread::

        exchange = fx.coop.BlockExchange[fx.Float32, 256, 4]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        striped = exchange.blocked_to_striped(values, storage=storage)

    ``values`` must be a thread-local :class:`~flydsl.expr.typing.Vector` with
    exactly ``items_per_thread`` elements of ``dtype``. The result is a flat
    vector with the same type and length.

    The static source and destination layouts select register-local, wave-local,
    or LDS lowering. Only LDS lowering requires every block thread to arrive
    together and performs a block barrier. Reusing LDS storage for another
    collective requires a barrier after the call because final reads have no
    trailing barrier.

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
    _blocked_to_striped_lowering = None
    _striped_to_blocked_lowering = None
    _blocked_to_warp_striped_lowering = None
    _warp_striped_to_blocked_lowering = None
    SharedStorage = None

    warp_ops = _dispatched_warp

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
    def blocked_to_striped(cls, values, *, storage=None):
        """Redistribute a blocked arrangement across the whole block."""
        return cls._exchange(
            values,
            _BLOCKED,
            _STRIPED,
            cls._blocked_to_striped_lowering,
            storage,
        )

    @classmethod
    def striped_to_blocked(cls, values, *, storage=None):
        """Redistribute a block-striped arrangement back to blocked."""
        return cls._exchange(
            values,
            _STRIPED,
            _BLOCKED,
            cls._striped_to_blocked_lowering,
            storage,
        )

    @classmethod
    def blocked_to_warp_striped(cls, values, *, storage=None):
        """Redistribute blocked items independently inside each logical warp."""
        return cls._exchange(
            values,
            _BLOCKED,
            _WARP_STRIPED,
            cls._blocked_to_warp_striped_lowering,
            storage,
        )

    @classmethod
    def warp_striped_to_blocked(cls, values, *, storage=None):
        """Redistribute warp-striped items back to block-wide blocked ownership."""
        return cls._exchange(
            values,
            _WARP_STRIPED,
            _BLOCKED,
            cls._warp_striped_to_blocked_lowering,
            storage,
        )

    @classmethod
    def _exchange(cls, values, source_name, destination_name, lowering, storage):
        if lowering == _REGISTER:
            return values
        if lowering == _WARP:
            return cls._exchange_warp(values, source_name, destination_name)
        return cls._exchange_lds(values, source_name, destination_name, storage)

    @classmethod
    def _exchange_warp(cls, values, source, destination):
        tid = linear_thread_id(cls.block_size)
        lane = tid % cls.warp_threads
        outputs = []
        for item in range(cls.items_per_thread):
            rank = cls._runtime_rank(destination, tid, lane, item)
            source_thread, source_item = cls._runtime_owner(source, rank)
            source_values = cls.warp_ops.warp_permute(
                values,
                source_thread % cls.warp_threads,
                width=cls.warp_threads,
            )
            outputs.append(source_values[source_item])
        return Vector.from_elements(outputs, cls.dtype)

    @classmethod
    def _exchange_lds(cls, values, source_name, destination_name, storage):
        tid = linear_thread_id(cls.block_size)
        warp_id = tid // cls.warp_threads
        lane = tid % cls.warp_threads
        source = cls._layout(source_name)
        destination = cls._layout(destination_name)
        for item in range(cls.items_per_thread):
            source_rank = cls._layout_rank(source, source_name, tid, warp_id, lane, item)
            storage.buffer[cls._storage_index(source_rank)] = values[item]
        barrier()
        return Vector.from_elements(
            [
                storage.buffer[
                    cls._storage_index(cls._layout_rank(destination, destination_name, tid, warp_id, lane, item))
                ]
                for item in range(cls.items_per_thread)
            ],
            cls.dtype,
        )

    @classmethod
    def _layout(cls, name):
        if name == _BLOCKED:
            return cls.blocked_layout()
        if name == _STRIPED:
            return cls.striped_layout()
        return cls.warp_striped_layout()

    @staticmethod
    def _layout_rank(layout, name, tid, warp_id, lane, item):
        coord = (warp_id, lane, item) if name == _WARP_STRIPED else (tid, item)
        return get_scalar(layout(*coord))

    @classmethod
    def _runtime_rank(cls, layout, tid, lane, item):
        if layout == _BLOCKED:
            return tid * cls.items_per_thread + item
        if layout == _STRIPED:
            return tid + item * cls.block_threads
        warp_id = tid // cls.warp_threads
        return warp_id * cls.warp_threads * cls.items_per_thread + lane + item * cls.warp_threads

    @classmethod
    def _runtime_owner(cls, layout, rank):
        if layout == _BLOCKED:
            return rank // cls.items_per_thread, rank % cls.items_per_thread
        if layout == _STRIPED:
            return rank % cls.block_threads, rank // cls.block_threads
        warp_rank = rank % (cls.warp_threads * cls.items_per_thread)
        warp_id = rank // (cls.warp_threads * cls.items_per_thread)
        return warp_id * cls.warp_threads + warp_rank % cls.warp_threads, warp_rank // cls.warp_threads

    @classmethod
    def _storage_index(cls, logical_index):
        """Map a logical rank into the padded LDS buffer.

        This performance-only mapping assumes 32 four-byte LDS banks. It
        preserves correctness on other bank geometries, but is not guaranteed
        to avoid their bank conflicts. Target metadata and an XOR-swizzled
        policy are intentionally left for a later implementation.
        """
        if cls.padding_items == 0:
            return logical_index
        period = _LDS_BANKS * cls.bank_items
        return logical_index + (logical_index // period) * cls.bank_items
