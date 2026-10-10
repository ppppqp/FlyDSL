#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Whole-register lane gathers, including packed tails and partial waves."""

import pytest
from coop_common import DTYPES, WARP_SIZE, dtype_id, sample

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

PERMUTE_DTYPES = (*DTYPES, (fx.Float16, "torch.float16"), (fx.BFloat16, "torch.bfloat16"))


def run_permute(values, dtype, *, width, items, universal=False, active=None, xor=False):
    block_threads = WARP_SIZE * 2
    active = block_threads if active is None else active

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid < active:
            namespace = fx.coop.universal if universal else fx.coop
            source = width - 1 if xor else width - 1 - tid % width
            permute = namespace.warp_permute_xor if xor else namespace.warp_permute
            if fx.const_expr(items == 0):
                result = permute(dtype(A[tid]), source, width=width)
                Out[tid] = result
            else:
                value = fx.Vector.from_elements([A[tid * items + i] for i in range(items)], dtype)
                result = permute(value, source, width=width)
                for i in range(items):
                    Out[tid * items + i] = result[i]

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(block_threads, 1, 1), stream=stream)

    out = torch.empty_like(values)
    launch(values, out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return out.cpu()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", PERMUTE_DTYPES, ids=dtype_id)
@pytest.mark.parametrize("items", (0, 1, 3, 4))
@pytest.mark.parametrize("universal", (False, True))
@pytest.mark.parametrize("xor", (False, True), ids=("indexed", "xor"))
def test_permute_preserves_values(entry, items, universal, xor):
    dtype, name = entry
    values = sample(name, WARP_SIZE * 2 * max(1, items))
    out = run_permute(values, dtype, width=WARP_SIZE, items=items, universal=universal, xor=xor)
    expected = values.cpu().reshape(2, WARP_SIZE, -1).flip(1).flatten()
    assert torch.equal(out, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width", (1, 2, 8, WARP_SIZE))
@pytest.mark.parametrize("universal", (False, True))
@pytest.mark.parametrize("xor", (False, True), ids=("indexed", "xor"))
def test_independent_logical_warps(width, universal, xor):
    values = sample("torch.int32", WARP_SIZE * 2 * 3)
    out = run_permute(values, fx.Int32, width=width, items=3, universal=universal, xor=xor)
    assert torch.equal(out, values.cpu().reshape(-1, width, 3).flip(1).flatten())


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_partial_physical_wave():
    # Three active logical groups; the rest of the physical wave is inactive.
    width = WARP_SIZE // 4
    active = 3 * width
    values = sample("torch.float16", active * 3)
    out = run_permute(values, fx.Float16, width=width, items=3, active=active)
    assert torch.equal(out, values.cpu().reshape(3, width, 3).flip(1).flatten())


@pytest.mark.l0_backend_agnostic
def test_invalid_arguments(ctx, insert_point):
    with pytest.raises(ValueError, match="power of two"):
        fx.coop.warp_permute(fx.Int32(1), 0, width=3)
    with pytest.raises(ValueError, match="source_lane"):
        fx.coop.warp_permute(fx.Int32(1), 4, width=4)
    with pytest.raises(TypeError, match="scalar or Vector"):
        fx.coop.warp_permute(1, 0)
