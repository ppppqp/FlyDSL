# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""One-output-per-block reduction with ``BlockReduce.reduce_to_leader``.

Every thread contributes a flag, but only linear thread 0 needs the aggregate:
it is the one thread that stores the block's count. The leader-only form avoids
forming the cross-warp result in threads that will not consume it.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK = 256


@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def count_positive(A: fx.Tensor, Count: fx.Tensor):
    tid = fx.thread_idx.x
    block_reduce = fx.coop.BlockReduce[fx.Int32, fx.known_block_size()]
    storage = fx.SharedAllocator().allocate(block_reduce.SharedStorage).peek()

    keep = (A[tid] > 0.0).select(1, 0)
    total = block_reduce.reduce_to_leader(keep, fx.ReductionOp.ADD, storage=storage)
    if tid == 0:
        Count[0] = total


@flyc.jit
def count(A: fx.Tensor, Count: fx.Tensor):
    count_positive(A, Count).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1))


A = torch.randn(BLOCK, dtype=torch.float32, device="cuda")
Count = torch.zeros(1, dtype=torch.int32, device="cuda")

count(A, Count)
torch.cuda.synchronize()

expected = int((A.cpu() > 0).sum())
if int(Count.cpu()[0]) == expected:
    print(f"PASS ({expected} positive elements)")
else:
    print("FAIL")
    raise SystemExit(1)
