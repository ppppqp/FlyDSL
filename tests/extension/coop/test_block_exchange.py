#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Block-wide exchange of register-held items.

Covered below: all three arrangements across the dtype list, item counts,
block shapes, and every legal block width. A round trip reuses shared storage
with the caller-provided barrier between exchanges.

``BlockExchange`` requires power-of-two block and item counts. Blocks narrower
than a wave use their block size as the logical warp width. Only flat Vectors
are accepted; arbitrary layouts and scatter/gather are not part of this API.
"""

from __future__ import annotations

import pytest
from coop_common import (
    BLOCK_THREADS,
    DTYPES,
    SUB_WARP_BLOCK_THREADS,
    WARP_SIZE,
    dtype_id,
    linear_tid,
    sample,
)

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None


CONVERSIONS = ("blocked_to_striped", "striped_to_blocked", "blocked_to_warp_striped", "warp_striped_to_blocked")
EXCHANGE_DTYPES = (*DTYPES, (fx.Float16, "torch.float16"))
BLOCK_SHAPES = ((64, 1, 1), (128, 1, 1), (256, 1, 1), (64, 2, 2))


def run_block_exchange(values, dtype, *, block_size, items_per_thread, conversion, universal=False, no_storage=False):
    """Exchange *values* once per thread; return the result on the host."""

    @flyc.kernel(known_block_size=list(block_size))
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        namespace = fx.coop.universal if universal else fx.coop
        exchange = namespace.BlockExchange[dtype, block_size, items_per_thread]
        storage = None if no_storage else fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        tid = linear_tid(block_size)
        base = tid * items_per_thread
        items = fx.Vector.from_elements([A[base + i] for i in range(items_per_thread)], dtype)
        out = getattr(exchange, conversion)(items, storage=storage)
        for i in range(items_per_thread):
            Out[base + i] = out[i]

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=block_size, stream=stream)

    out = torch.empty_like(values)
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return out.cpu()


def check_exchange(values, out, *, block_threads, items_per_thread, conversion):
    """Compare each arrangement with a host tensor transpose, exactly."""
    host = values.cpu()
    if conversion == "blocked_to_striped":
        expected = host.reshape(items_per_thread, block_threads).T.flatten()
    elif conversion == "striped_to_blocked":
        expected = host.reshape(block_threads, items_per_thread).T.flatten()
    else:
        width = min(WARP_SIZE, block_threads)
        if conversion == "blocked_to_warp_striped":
            expected = host.reshape(-1, items_per_thread, width).transpose(1, 2).flatten()
        else:
            expected = host.reshape(-1, width, items_per_thread).transpose(1, 2).flatten()
    assert torch.equal(out, expected)


# ── named arrangements ────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", EXCHANGE_DTYPES, ids=dtype_id)
@pytest.mark.parametrize("items_per_thread", (1, 4, 8))
@pytest.mark.parametrize("conversion", CONVERSIONS)
def test_exchange_over_a_full_tile(entry, items_per_thread, conversion):
    """Each dtype preserves every value through the named permutation."""
    dtype, name = entry
    block_size = (128, 1, 1)
    values = sample(name, 128 * items_per_thread)

    out = run_block_exchange(
        values, dtype, block_size=block_size, items_per_thread=items_per_thread, conversion=conversion
    )
    check_exchange(values, out, block_threads=128, items_per_thread=items_per_thread, conversion=conversion)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_size", BLOCK_SHAPES, ids=lambda b: "x".join(map(str, b)))
@pytest.mark.parametrize("conversion", CONVERSIONS)
def test_exchange_across_block_shapes(block_size, conversion):
    """Multidimensional blocks use the same x-fast thread order."""
    items_per_thread = 4
    block_threads = block_size[0] * block_size[1] * block_size[2]
    values = sample("torch.int32", block_threads * items_per_thread)

    out = run_block_exchange(
        values, fx.Int32, block_size=block_size, items_per_thread=items_per_thread, conversion=conversion
    )
    check_exchange(values, out, block_threads=block_threads, items_per_thread=items_per_thread, conversion=conversion)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_threads", BLOCK_THREADS, ids=lambda n: f"t{n}")
@pytest.mark.parametrize("conversion", CONVERSIONS)
def test_exchange_across_every_legal_block_width(block_threads, conversion):
    """One thread through 1024, including partial waves and multiple waves."""
    items_per_thread = 4
    values = sample("torch.int32", block_threads * items_per_thread)

    out = run_block_exchange(
        values,
        fx.Int32,
        block_size=(block_threads, 1, 1),
        items_per_thread=items_per_thread,
        conversion=conversion,
    )
    check_exchange(values, out, block_threads=block_threads, items_per_thread=items_per_thread, conversion=conversion)


# ── storage reuse ─────────────────────────────────────────────────────────


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block_size", ((WARP_SIZE // 2, 1, 1), (64, 2, 2)), ids=lambda b: "x".join(map(str, b)))
def test_round_trip_reuses_shared_storage(block_size):
    """A barrier lets the inverse overwrite the first exchange's scratch."""
    ITEMS = 4
    block_threads = block_size[0] * block_size[1] * block_size[2]

    @flyc.kernel(known_block_size=list(block_size))
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        exchange = fx.coop.BlockExchange[fx.Int32, block_size, ITEMS]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        base = linear_tid(block_size) * ITEMS
        values = fx.Vector.from_elements([A[base + i] for i in range(ITEMS)])
        striped = exchange.blocked_to_striped(values, storage=storage)
        fx.barrier()
        blocked = exchange.striped_to_blocked(striped, storage=storage)
        for i in range(ITEMS):
            Out[base + i] = blocked[i]

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=block_size, stream=stream)

    values = sample("torch.int32", block_threads * ITEMS)
    out = torch.empty_like(values)
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), values.cpu())


# ── specialization and input contract ──────────────────────────────────────


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("block_threads", SUB_WARP_BLOCK_THREADS, ids=lambda n: f"t{n}")
def test_a_sub_wave_block_narrows_its_logical_warp(block_threads):
    exchange = fx.coop.BlockExchange[fx.Int32, block_threads, 4]
    assert exchange.warp_threads == block_threads


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "params",
    ((fx.Int32, 3, 4), (fx.Int32, 64, 3), (fx.Int32, 2048, 2), (fx.Int32, 64, 0), (fx.Int32, 64, True), (int, 64, 4)),
)
def test_invalid_specializations(params):
    with pytest.raises((TypeError, ValueError)):
        fx.coop.BlockExchange[params]


@pytest.mark.l0_backend_agnostic
def test_value_diagnostics(ctx, insert_point):
    exchange = fx.coop.BlockExchange[fx.Int32, 64, 4]
    assert exchange is fx.coop.BlockExchange[fx.Int32, (64, 1, 1), 4]
    assert issubclass(fx.coop.universal.BlockExchange, fx.coop.BlockExchange)
    with pytest.raises(TypeError, match="already specialized"):
        exchange[fx.Int32, 64, 4]
    with pytest.raises(TypeError, match="specialize first"):
        fx.coop.BlockExchange.blocked_to_striped(None, storage=None)
    with pytest.raises(TypeError, match="must be a Vector"):
        exchange.blocked_to_striped(fx.Int32(1), storage=None)
    wrong_type = fx.Vector.from_elements([fx.Float32(1)] * 4)
    with pytest.raises(TypeError, match="expects Int32"):
        exchange.blocked_to_striped(wrong_type, storage=None)
    short = fx.Vector.from_elements([fx.Int32(1)] * 2)
    with pytest.raises(ValueError, match="flat Vector of 4"):
        exchange.blocked_to_striped(short, storage=None)
    values = fx.Vector.from_elements([fx.Int32(1)] * 4)
    with pytest.raises(ValueError, match="flat Vector"):
        exchange.blocked_to_striped(values.reshape((2, 2)), storage=None)
    with pytest.raises(TypeError, match="requires SharedStorage"):
        exchange.blocked_to_striped(values, storage=None)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("conversion", CONVERSIONS)
@pytest.mark.parametrize("block_threads", (1, WARP_SIZE // 2, WARP_SIZE))
@pytest.mark.parametrize("universal", (False, True))
def test_wave_local_exchange_needs_no_storage(conversion, block_threads, universal):
    values = sample("torch.int32", block_threads * 4)
    out = run_block_exchange(
        values,
        fx.Int32,
        block_size=(block_threads, 1, 1),
        items_per_thread=4,
        conversion=conversion,
        universal=universal,
        no_storage=True,
    )
    check_exchange(values, out, block_threads=block_threads, items_per_thread=4, conversion=conversion)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("conversion", ("blocked_to_warp_striped", "warp_striped_to_blocked"))
def test_warp_striping_in_multiple_waves_needs_no_storage(conversion):
    values = sample("torch.int32", 256 * 4)
    out = run_block_exchange(
        values, fx.Int32, block_size=(64, 2, 2), items_per_thread=4, conversion=conversion, no_storage=True
    )
    check_exchange(values, out, block_threads=256, items_per_thread=4, conversion=conversion)
