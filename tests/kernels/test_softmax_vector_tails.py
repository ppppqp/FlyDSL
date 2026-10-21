# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Softmax row boundaries, unaligned row starts, and partial vector tiles."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.norm.softmax_bwd_kernel import build_softmax_bwd_module
from kernels.norm.softmax_common import _load_vec, _store_vec
from kernels.norm.softmax_kernel import build_softmax_module

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@pytest.mark.parametrize(
    "dtype_str,dtype,tol", [("f32", torch.float32, 1e-5), ("f16", torch.float16, 2e-3), ("bf16", torch.bfloat16, 1e-2)]
)
@pytest.mark.parametrize("N", [1, 3, 7, 8, 9, 127, 255, 256, 257, 1000, 1023, 1025, 2000, 2047, 2049, 4097])
def test_softmax_vector_row_boundaries(dtype_str, dtype, tol, N):
    # Odd N shifts each row's alignment; guards expose stores past the tensor.
    M = 7
    torch.manual_seed(N)
    x = torch.randn((M, N), device="cuda", dtype=dtype)
    dy = torch.randn_like(x)
    storage = torch.full((M * N + 32,), -123, device="cuda", dtype=dtype)
    out = storage[16:-16].view(M, N)
    y = torch.softmax(x.float(), dim=1).to(dtype)
    references = [
        torch.softmax(x.float(), dim=1),
        y.float() * (dy.float() - (dy.float() * y.float()).sum(1, keepdim=True)),
    ]
    # Include a partial final multi-row block and a subgroup narrower than a wave.
    for threads_per_row, rows_per_block in [(256, 1), (8, 16)]:
        forward = build_softmax_module(
            M,
            N,
            dtype_str,
            BLOCK_THREADS=threads_per_row * rows_per_block,
            THREADS_PER_ROW=threads_per_row,
            ROWS_PER_BLOCK=rows_per_block,
        )
        forward(x, out, M)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), references[0], atol=tol * references[0].abs().max().item(), rtol=0)
        assert bool((storage[:16] == -123).all() & (storage[-16:] == -123).all())
    backward = build_softmax_bwd_module(N, dtype_str)
    backward(dy, y, out, M)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), references[1], atol=tol * references[1].abs().max().item(), rtol=0)
    assert bool((storage[:16] == -123).all() & (storage[-16:] == -123).all())


@pytest.mark.parametrize("N", [16383, 16385, 32767, 32769, 65537])
@pytest.mark.parametrize("dtype_str,dtype,tol", [("f32", torch.float32, 1e-5), ("bf16", torch.bfloat16, 1e-2)])
def test_softmax_backward_wide_tail(N, dtype_str, dtype, tol):
    """Reloaded operands must use safe vector indices past the residency caps."""
    M = 3
    torch.manual_seed(N)
    y = torch.randn((M, N), device="cuda").softmax(1).to(dtype)
    dy = torch.randn_like(y)
    dx = torch.empty_like(y)
    build_softmax_bwd_module(N, dtype_str)(dy, y, dx, M)
    reference = y.float() * (dy.float() - (dy.float() * y.float()).sum(1, keepdim=True))
    torch.testing.assert_close(dx.float(), reference, atol=tol * reference.abs().max().item(), rtol=0)


@pytest.mark.parametrize("dtype_str,dtype", [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)])
@pytest.mark.parametrize("scalar", [False, True])
def test_masked_copy_contract(dtype_str, dtype, scalar):
    """Check valid loads, safe inactive addresses, and suppressed stores."""
    width = 1 if scalar else (4 if dtype_str == "f32" else 8)
    source = torch.arange(3 * width, device="cuda", dtype=torch.float32).to(dtype).view(1, -1)
    output = torch.empty((1, 32 * width), device="cuda", dtype=dtype)
    guarded = torch.full_like(output, 123)

    @flyc.kernel
    def copy_kernel(source: fx.Tensor, output: fx.Tensor, guarded: fx.Tensor):
        tid = fx.thread_idx.x
        src = fx.slice(fx.rocdl.make_buffer_tensor(source), (0, None))
        dst = fx.slice(fx.rocdl.make_buffer_tensor(output), (0, None))
        guard = fx.slice(fx.rocdl.make_buffer_tensor(guarded), (0, None))
        value = _load_vec(src, tid, width, dtype_str, mask=tid < 3)
        _store_vec(dst, tid, value, width, dtype_str, mask=tid < 32)
        _store_vec(guard, tid, value, width, dtype_str, mask=tid < 3)

    @flyc.jit
    def launch(source: fx.Tensor, output: fx.Tensor, guarded: fx.Tensor):
        copy_kernel(source, output, guarded).launch(grid=(1, 1, 1), block=(32, 1, 1))

    launch(source, output, guarded)
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:, : 3 * width], source, rtol=0, atol=0)
    torch.testing.assert_close(guarded[:, : 3 * width], source, rtol=0, atol=0)
    assert bool((guarded[:, 3 * width :] == 123).all())
