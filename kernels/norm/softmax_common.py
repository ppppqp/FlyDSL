# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Masked row copies shared by softmax forward and backward."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.common.kernels_common import dtype_to_elem_type


def row_copy_chunks(N, threads_per_row, vec_width):
    """Return (copy width, starting copy index, exclusive copy limit) tuples.

    Complete vectors use 128-bit copies. A final width-one chunk distributes
    the remaining scalar elements across the first few row threads.
    """
    vectors = N // vec_width
    chunks = [(vec_width, base, vectors) for base in range(0, vectors, threads_per_row)]
    if N % vec_width:
        chunks.append((1, vectors * vec_width, N))
    return tuple(chunks)


@flyc.jit
def _load_vec(row, index, width: fx.Constexpr[int], dtype_str: fx.Constexpr[str], mask):
    """Load a complete copy unit with safe addresses for every lane.

    Masked values are unspecified; consumers must mask their results.
    """
    dtype = dtype_to_elem_type(dtype_str)
    bits = width * (32 if dtype_str == "f32" else 16)
    atom_type = (
        fx.rocdl.BufferCopy128b if bits == 128 else (fx.rocdl.BufferCopy32b if bits == 32 else fx.rocdl.BufferCopy16b)
    )
    atom = fx.make_copy_atom(atom_type(), 32 if dtype_str == "f32" else 16)
    divided = fx.logical_divide(row, fx.make_layout(width, 1))
    registers = fx.make_rmem_tensor(width, dtype)
    # A false mask must never issue an out-of-bounds load, even if its result
    # will be discarded. Reading copy zero keeps the address safe.
    safe_index = mask.select(index, 0)
    fx.copy(atom, fx.slice(divided, (None, safe_index)), registers)
    return fx.memref_load_vec(registers)


@flyc.jit
def _store_vec(row, index, value, width: fx.Constexpr[int], dtype_str: fx.Constexpr[str], mask):
    """Store a complete copy unit only for lanes enabled by ``mask``."""
    dtype = dtype_to_elem_type(dtype_str)
    bits = width * (32 if dtype_str == "f32" else 16)
    atom_type = (
        fx.rocdl.BufferCopy128b if bits == 128 else (fx.rocdl.BufferCopy32b if bits == 32 else fx.rocdl.BufferCopy16b)
    )
    atom = fx.make_copy_atom(atom_type(), 32 if dtype_str == "f32" else 16)
    divided = fx.logical_divide(row, fx.make_layout(width, 1))
    registers = fx.make_rmem_tensor(width, dtype)
    fx.memref_store_vec(value.to(dtype), registers)
    if mask:
        fx.copy(atom, registers, fx.slice(divided, (None, index)))
