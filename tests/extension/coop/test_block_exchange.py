#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Portable block exchange and its named thread/value layouts."""

from __future__ import annotations

import pytest
from coop_common import BLOCK_THREADS, DTYPES, WARP_SIZE, dtype_id, linear_tid, torch_dtype

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler import jit_function
from flydsl.compiler.backends.rocm import RocmBackend

try:
    import torch
except ImportError:
    torch = None


ITEM_COUNTS = (1, 2, 4, 8)


def _blocked_rank(thread, item, block_threads, items_per_thread):
    """Reference ``(thread, item) -> logical rank`` for a blocked layout."""
    del block_threads
    return thread * items_per_thread + item


def _striped_rank(thread, item, block_threads, items_per_thread):
    """Reference ``(thread, item) -> logical rank`` for a striped layout."""
    del items_per_thread
    return thread + item * block_threads


def _warp_striped_rank(thread, item, warp_threads, items_per_thread):
    """Reference logical rank for a layout striped independently per warp."""
    warp_id, lane = divmod(thread, warp_threads)
    return warp_id * warp_threads * items_per_thread + lane + item * warp_threads


@pytest.fixture
def frontend_only_compile(monkeypatch):
    """Trace exchange kernels without requiring a GPU or binary generation."""
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))


@pytest.fixture(params=("gfx942", "gfx1201"))
def pre_binary_compile(request, monkeypatch):
    """Run wave64 and wave32 lowering, stopping before code-object generation."""
    arch = request.param
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")
    lowered = []

    def compile_pre_binary(cls, module, **_kwargs):
        backend = RocmBackend(RocmBackend.make_target(arch))
        module = ir.Module.parse(module.operation.get_asm())
        pipeline, _ = backend.external_binary_pipeline_fragments(compile_hints={})
        PassManager.parse(f"builtin.module({','.join(pipeline)})", module.context).run(module.operation)
        assert module.operation.verify()
        lowered.append(str(module))
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_pre_binary))
    return arch, lowered


def _physical_ranks(layout, block_threads, items_per_thread, warp_threads):
    rank = {
        "blocked": lambda thread, item: _blocked_rank(thread, item, block_threads, items_per_thread),
        "striped": lambda thread, item: _striped_rank(thread, item, block_threads, items_per_thread),
        "warp_striped": lambda thread, item: _warp_striped_rank(thread, item, warp_threads, items_per_thread),
    }[layout]
    return [rank(thread, item) for thread in range(block_threads) for item in range(items_per_thread)]


# ── host-side layout and specialization checks ────────────────────────────


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("block_threads", BLOCK_THREADS, ids=lambda n: f"t{n}")
@pytest.mark.parametrize("items_per_thread", ITEM_COUNTS, ids=lambda n: f"i{n}")
def test_every_named_layout_is_a_bijection(block_threads, items_per_thread):
    """Exhaust every supported launch width and item count without a GPU."""
    total = block_threads * items_per_thread
    warp_threads = min(WARP_SIZE, block_threads)
    expected = list(range(total))

    for layout in ("blocked", "striped", "warp_striped"):
        assert sorted(_physical_ranks(layout, block_threads, items_per_thread, warp_threads)) == expected


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("block_threads", BLOCK_THREADS, ids=lambda n: f"t{n}")
@pytest.mark.parametrize("items_per_thread", ITEM_COUNTS, ids=lambda n: f"i{n}")
def test_named_exchanges_preserve_logical_ranks(block_threads, items_per_thread):
    """The destination registers receive the same logical items under a new layout."""
    warp_threads = min(WARP_SIZE, block_threads)
    blocked = _physical_ranks("blocked", block_threads, items_per_thread, warp_threads)
    striped = _physical_ranks("striped", block_threads, items_per_thread, warp_threads)
    warp_striped = _physical_ranks("warp_striped", block_threads, items_per_thread, warp_threads)

    assert blocked == list(range(block_threads * items_per_thread))
    for source, destination in (
        (blocked, striped),
        (striped, blocked),
        (blocked, warp_striped),
        (warp_striped, blocked),
    ):
        # Map each source register to the destination register that owns the
        # same logical rank. It must itself be a complete permutation.
        movement = [destination.index(rank) for rank in source]
        assert sorted(movement) == blocked
        assert [source[source.index(rank)] for rank in destination] == destination


@pytest.mark.l1a_compile_no_target_dialect
def test_specialization_and_padding_metadata():
    exchange = fx.coop.BlockExchange[fx.Uint8, (64, 2, 1), 4]

    assert exchange is fx.coop.BlockExchange[fx.Uint8, (64, 2, 1), 4]
    assert exchange.block_size == (64, 2, 1)
    assert exchange.block_threads == 128
    assert exchange.items_per_thread == 4
    assert exchange.warp_threads == min(WARP_SIZE, 128)
    assert exchange.bank_items == 4
    assert exchange.padding_items == 16
    assert exchange.storage_items == 528
    assert exchange.SharedStorage.__dsl_field_defs__[0].type_spec.size == 528
    assert fx.coop.universal.BlockExchange is fx.coop.BlockExchange


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize(
    "params, error, message",
    [
        ((fx.Int32, 64), TypeError, "dtype, block_size, items_per_thread"),
        ((object, 64, 4), TypeError, "Numeric subclass"),
        ((fx.Boolean, 64, 4), TypeError, "byte-addressable"),
        ((fx.Int32, 48, 4), ValueError, "power of two"),
        ((fx.Int32, 64, 3), ValueError, "power of two"),
    ],
)
def test_invalid_specializations_are_rejected(params, error, message):
    with pytest.raises(error, match=message):
        fx.coop.BlockExchange[params]


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize(
    "method",
    (
        "blocked_to_striped",
        "striped_to_blocked",
        "blocked_to_warp_striped",
        "warp_striped_to_blocked",
    ),
)
def test_every_exchange_path_traces_to_valid_frontend_ir(method, frontend_only_compile):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel():
        exchange = fx.coop.BlockExchange[fx.Int32, 64, 4]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        values = fx.Vector.from_elements([fx.Int32(i) for i in range(4)])
        getattr(exchange, method)(values, storage=storage)

    @flyc.jit
    def launch():
        kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch()
    ir_text = launch._last_compiled[1].source_ir
    assert "gpu.barrier" in ir_text
    assert "fly.make_layout" in ir_text
    assert "vector.from_elements" in ir_text


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch tensor arguments")
def test_exchange_lowers_through_the_pre_binary_pipeline(pre_binary_compile):
    arch, lowered_modules = pre_binary_compile

    @flyc.kernel(known_block_size=[128, 1, 1])
    def kernel(Out: fx.Tensor):
        exchange = fx.coop.BlockExchange[fx.Int32, 128, 4]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        values = fx.Vector.from_elements([fx.Int32(i) for i in range(4)])
        values = exchange.blocked_to_striped(values, storage=storage)
        base = fx.thread_idx.x * 4
        for i in range(4):
            Out[base + i] = values[i]

    @flyc.jit
    def launch(Out: fx.Tensor):
        kernel(Out).launch(grid=(1, 1, 1), block=(128, 1, 1))

    launch(torch.empty(128 * 4, dtype=torch.int32))
    assert len(lowered_modules) == 1
    lowered = lowered_modules[0]
    assert "llvm.func" in lowered
    assert f'chip = "{arch}"' in lowered
    assert "llvm.load" in lowered
    assert "gpu.barrier" not in lowered
    assert "fly.make_layout" not in lowered


# ── actual device exchange ────────────────────────────────────────────────


def _run_exchange(values, dtype, *, block_size, items_per_thread, method):
    @flyc.kernel(known_block_size=list(block_size))
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        exchange = fx.coop.BlockExchange[dtype, block_size, items_per_thread]
        storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()
        tid = linear_tid(block_size)
        base = tid * items_per_thread
        items = fx.Vector.from_elements([A[base + i] for i in range(items_per_thread)])
        out = getattr(exchange, method)(items, storage=storage)
        for i in range(items_per_thread):
            Out[base + i] = out[i]

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=block_size, stream=stream)

    out = torch.empty_like(values)
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return out.cpu()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "method, source_layout, destination_layout",
    [
        ("blocked_to_striped", "blocked", "striped"),
        ("striped_to_blocked", "striped", "blocked"),
        ("blocked_to_warp_striped", "blocked", "warp_striped"),
        ("warp_striped_to_blocked", "warp_striped", "blocked"),
    ],
)
@pytest.mark.parametrize("block_size", ((32, 1, 1), (64, 2, 1)), ids=lambda b: "x".join(map(str, b)))
@pytest.mark.parametrize("items_per_thread", (1, 4), ids=lambda n: f"i{n}")
def test_exchange_permutations(method, source_layout, destination_layout, block_size, items_per_thread):
    block_threads = block_size[0] * block_size[1] * block_size[2]
    warp_threads = min(WARP_SIZE, block_threads)
    source = _physical_ranks(source_layout, block_threads, items_per_thread, warp_threads)
    destination = _physical_ranks(destination_layout, block_threads, items_per_thread, warp_threads)
    values = torch.tensor(source, dtype=torch.int32, device="cuda")

    out = _run_exchange(
        values,
        fx.Int32,
        block_size=block_size,
        items_per_thread=items_per_thread,
        method=method,
    )

    assert torch.equal(out, torch.tensor(destination, dtype=torch.int32))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
def test_blocked_to_striped_preserves_every_supported_dtype(entry):
    dtype, name = entry
    block_size = (64, 1, 1)
    items_per_thread = 4
    block_threads = block_size[0]
    source = _physical_ranks("blocked", block_threads, items_per_thread, min(WARP_SIZE, block_threads))
    destination = _physical_ranks("striped", block_threads, items_per_thread, min(WARP_SIZE, block_threads))
    values = torch.tensor(source, dtype=torch_dtype(name), device="cuda")

    out = _run_exchange(
        values,
        dtype,
        block_size=block_size,
        items_per_thread=items_per_thread,
        method="blocked_to_striped",
    )

    assert torch.equal(out, torch.tensor(destination, dtype=torch_dtype(name)))
