# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Vectorized, predicated, target-neutral 2D elementwise add (C = A + B).

This example is **target-neutral**: it uses only the backend-agnostic ``flydsl.expr`` API, so it
supports on any backend.

Highlights:
  1. **float4 vectorization** via ``UniversalCopy128b`` -- each copy atom moves 128 bits
     (4 x f32) along the contiguous (N) axis, so every thread loads/stores one ``float4``.
  2. **Predicated OOB masking**: the (M, N) shape need not be a multiple of the block tile,
     so border blocks have threads whose float4 lies past the tensor. A per-atom boolean
     predicate (``coord < (M, N)``) gates each copy, so a load/store never touches OOB memory.
"""

"""
NOTE:
1. TileCopy
TileCopy is built with `make_copy_atom` and `make_tiled_copy_tv`.
- Copy atom: one operation transfers 4 * float32
- Thread layout: 128 threads arranged as 8 * 16 grid
- Value layout: each thread has 1 * 4 values
Thread layout is row-major, so shape = (8, 16) and stride = (16, 1)
Therefore, thread_m = tid // 16, thread_n = tid % 16
For example, tid = 15 => thread coordinate = (0, 15) => row 0, columns 60-63
             tid = 16 => thread coordinate = (1, 0) => row 1, columns 0-3
See `derived.py` for details on make_layout_tv

2. flat_divide
For a tensor view A[m, n], with shape = (M, N),
fx.flat_divide(A, (8, 64)) returns a view with shape = (8, 64, M//8, N//64)
- m = tile_m_index * 8  + intra_m
- n = tile_n_index * 64 + intra_n
So it creates a logical mode of (intra_m, intra_n, tile_m_index, tile_n_index)
with shape (8, 64, ceil(M / 8), ceil(N / 64))
Therefore, fx.flat_divide(A, (8, 64))[None, None, bid_x, bid_y] means keep all intra_m and intra_n coordinates, get bid_x and bid_y tile
See `Fly/Utils/LayoutUtils.h` for layoutFlatDivide
the python flat_divide basically convert (8, 64) into a Fly Tile, emits fly.flat_divide and returns a tensor-like value with inferred layout

3. get_slice
It's a thin wrapper that returns a ThrCopy object, which is a per-thread view of the TiledCopy. It provides partition_S, partition_D, and retile methods for tensor partitioning.
class ThrCopy:
    def __init__(self, tiled_copy, tid):
        self.tiled_copy = tiled_copy
        self.tid = tid

    def partition_S(self, tensor):
        return make_thread_view(
            tensor,
            thread=self.tid,
            mapping=self.tiled_copy.source_mapping,
        )

    def partition_D(self, tensor):
        return make_thread_view(
            tensor,
            thread=self.tid,
            mapping=self.tiled_copy.destination_mapping,
        )

4. partition_S and partition_D
It emits fly.tiled_copy_partition_src %tiled_copy, %src, %tid
thr_gA = thr_copy.partition_S(gA) is approximately
- thread_m = tid // 16
- thread_n = tid % 16
thr_gA[v] <-> gA[thread_m, thread_n * 4 + v], for v = 0, 1, 2, 3
where thread_m = bid_x * 8 + tid // 16, bid_y * 64 + (tid % 16) * 4 + v
Basically mapping to thread local index


Mental model:
  flat_divide:
      choose which tile the block owns

  get_slice:
      remember which thread is asking

  partition_S/D:
      calculate which addresses inside that tile the thread own
"""


import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def vector_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    tiled_copy: fx.TiledCopy,  # NOTE: a compile-time description of how threads collectively copy a tile
    # it encodes the mapping between threads, vector values and tensor coordinates.
):
    tid = fx.thread_idx.x
    bid_x, bid_y = fx.block_idx.x, fx.block_idx.y

    # NOTE: bid_x selects a tile along M
    # bid_y selects a tile along N
    # tid determines wich portion of the tile belongs to the current thread

    # Identity (coordinate) tensor: value == logical coord (m, n).
    M, N = A.shape.unpack()
    idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))
    # NOTE: idC is not an allocated data tensor. It is a symbolic coordinate tensor whose value at each position is that position's coordinate
    # idC[m, n] == (m, n)
    # The kernel later partitions this coordinate tensor exactly like the data tensors. That tells each thread which global coordinates its vector load corresponds to.
    # Those coordinates are used to construct the out-of-bounds predicate.

    TileMN = tiled_copy.tile_mn
    # NOTE: The JIT wrapper constructs tiled_copy using:
    # - a thread layout of (8, 16)
    # - a tile layout of (1, 4)
    # together they form a block tile of TileMN = (8, 16) * (1, 4) = (8, 64)
    # in other words, 8 thread positions along M, 16 thread positions along N, each thread loads a float4 (4 contiguous f32) along N, so the block tile is 8 x 64.

    gA = fx.flat_divide(A, TileMN)[None, None, bid_x, bid_y]
    gB = fx.flat_divide(B, TileMN)[None, None, bid_x, bid_y]
    gC = fx.flat_divide(C, TileMN)[None, None, bid_x, bid_y]
    cC = fx.flat_divide(idC, TileMN)[None, None, bid_x, bid_y]
    # NOTE: selecting this block's tile.
    # flat_divide logically divides each tensor into (8, 64) tiles. Conceptually, it changes the view from:
    # - tensor[m, n]
    # to something like
    # - tensor[in_tile_m, in_tile_n, tile_m_index, tile_n_index]
    # then, [None, None, bid_x, bid_y] selects the current block's tile while retaining all coordinates inside that tile
    # gA, gB, gC are the data tensors for this block's tile'
    # g stands for "global", c stands for "coordinate"

    thr_copy = tiled_copy.get_slice(tid)
    # NOTE: tiled_copy describes the block-level copy. get_slice(tid) specialize the mapping for the current thread

    thr_gA = thr_copy.partition_S(gA)
    thr_gB = thr_copy.partition_S(gB)
    thr_gC = thr_copy.partition_D(gC)
    thr_cC = thr_copy.partition_S(cC)[(0, None), None, None]
    # NOTE: partition_S: partition according to the copy atom's source mapping.
    # partition_D: partition according to the copy atom's destination mapping.
    # for C, global memory is the destination. The additional indexing removes or selects layout modes that
    # are not needed for predicate generation. The resulting tensor has one relevant coordinate per copy atom
    # rather than redundantly retaining every vector lane’s coordinate.

    thr_rA = fx.make_fragment_like(thr_gA)
    thr_rB = fx.make_fragment_like(thr_gB)
    thr_rC = fx.make_fragment_like(thr_gC)
    thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)
    # NOTE: These create local fragments matching the layout of the thread partitioned views.
    # thr_rA register holds the four loaded A values
    # thr_rC resgister holds the four results
    # thr_pC are boolean registers containing validity predicates

    for a in fx.range_constexpr(fx.size(thr_pC.shape).unpack()):
        # NOTE: range_constexpr is a compile-time loop. Unrolled while tracing
        # for each copy atom, computes element wise coordinate comparison
        thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))

    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    # NOTE: creates a copy atom that moves 128 bits (4 x f32) along the contiguous axis.

    fx.copy(copy_atom, thr_gA, thr_rA, pred=thr_pC)
    fx.copy(copy_atom, thr_gB, thr_rB, pred=thr_pC)
    # NOTE: These operations copy data from global-memory views into register fragments.

    thr_rC.store(thr_rA.load() + thr_rB.load())
    # NOTE: perform addition.

    fx.copy(copy_atom, thr_rC, thr_gC, pred=thr_pC)
    # NOTE: This operation copies the results from the register fragments back to the global-memory views.


@flyc.jit
def vector_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    tiled_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_ordered_layout((8, 16), order=(1, 0)),
        fx.make_ordered_layout((1, 4), order=(0, 1)),
    )
    # NOTE: core mapping decision.
    # thread layuout: fx.make_ordered_layout((8, 16), order=(1, 0)), arranges 128 threads as a 8*16 grid
    # per-thread value layout layout: fx.make_ordered_layout((1, 4), order=(0, 1)) assigns each thread one row by 4 contiguous columns

    tile_m, tile_n = tiled_copy.tile_mn.unpack()

    M, N = A.shape.unpack()
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n
    vector_add_kernel(A, B, C, tiled_copy).launch(grid=(grid_m, grid_n, 1), block=(8 * 16, 1, 1), stream=stream)


M, N = 100, 1000

A = torch.randn(M, N, dtype=torch.float32, device=torch.device("cuda"))
B = torch.randn(M, N, dtype=torch.float32, device=torch.device("cuda"))
C = torch.zeros(M, N, dtype=torch.float32, device=torch.device("cuda"))

vector_add(A, B, C, stream=torch.cuda.Stream())
torch.cuda.synchronize()

if torch.allclose(A + B, C):
    print("PASS")
else:
    print("FAIL:")
    print(A + B)
    print(C)
