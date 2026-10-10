# Proposal: Layout-Directed Cooperative Tile Redistribution

## Recommendation

FlyDSL should gain a reusable mechanism for redistributing register-held tiles between thread/value layouts—initially as `fx.coop.BlockExchange`, and eventually as a general layout-conversion primitive.

This fills a gap between FlyDSL's existing strengths:

- The Fly dialect precisely describes memory and thread/value layouts.
- MMA and copy atoms expose the layouts they require.
- A loaded `Vector`, however, retains only its thread-local shape; it does not say which logical elements its registers represent across the block.
- Consequently, production kernels repeatedly encode redistribution using `ds_bpermute`, DPP, register packing, and LDS.

Representative examples include the hand-written rotary-pair exchange in `kernels/attention/fused_rope_cache_kernel.py` and the wave prefix/permutation code in `kernels/moe/moe_sorting_kernel.py`.

The direction also matches the maintainers' request to grow `flydsl.extension` with reusable cooperative algorithms, including striped/blocked exchange, in [FlyDSL issue #1016](https://github.com/ROCm/FlyDSL/issues/1016).

## Concrete First Contribution

Implement a block-exchange interface along these lines:

```python
exchange = fx.coop.BlockExchange[
    dtype,
    block_size,
    items_per_thread,
]

striped = exchange.blocked_to_striped(values, storage=storage)
blocked = exchange.striped_to_blocked(values, storage=storage)
warp_striped = exchange.blocked_to_warp_striped(values, storage=storage)
```

Here, `values` is a thread-local `Vector`. The initial implementation can use padded LDS, following the model of rocPRIM's block exchange, while representing the arrangements internally as static Fly layouts:

```text
blocked: logical = thread * items_per_thread + item
striped: logical = thread + item * block_threads
```

rocPRIM supports these transformations, as well as scatter/gather variants, and pads temporary storage to avoid bank conflicts. It provides a mature semantic reference without requiring FlyDSL to wrap rocPRIM itself. See the [rocPRIM BlockExchange documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.2.1/block_ops/ops_classes/exchange.html).

## Layout-Directed Lowering

The feature becomes more than a direct rocPRIM port if it includes a small compile-time planner that compares the source and destination thread/value layouts:

| Conversion locality | Lowering strategy |
|---|---|
| Same owning thread | `vector.shuffle`, extract, and insert operations |
| Different lane in the same wave | DPP, `readlane`, or `ds_bpermute` |
| Different wave in the same block | Padded LDS and block barriers |
| Unsupported or non-bijective mapping | Compile-time diagnostic |

The first implementation should be restricted to static, bijective, power-of-two layouts and named BlockExchange arrangements. It should not start with a fully general, dynamic `fly.layout_convert` operation; that would make the initial correctness and review surface unnecessarily large.

Once the planner has multiple real users, it could support a more general interface:

```python
result = fx.coop.redistribute(
    values,
    source=source_thread_value_layout,
    destination=destination_thread_value_layout,
)
```

## Related Work

Triton's layout system explicitly models register, lane, warp, and block dimensions and uses them for generic layout-to-layout conversion. Its Linear Layout work argues that systematic layout representations can replace a growing collection of case-specific conversion rules:

- [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2](https://arxiv.org/abs/2505.23819)
- [Triton LinearLayout conversion interface](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h)

Triton's scan lowering decomposes a distributed operation into thread-local work, wave shuffles, and cross-wave shared-memory exchange. The local implementation is in `/home/qiping-pan/Documents/workspace/triton/lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp`.

TileLang exposes tile operations at a higher level, but its current fragment scan stages through shared memory. The local implementation is in `/home/qiping-pan/Documents/workspace/tilelang/tilelang/language/scan_op.py`. A Fly layout-aware planner could preserve FlyDSL's explicit control while avoiding LDS when the permutation is thread- or wave-local.

The 2026 [Axe layout paper](https://arxiv.org/abs/2601.19092) provides further motivation for connecting hardware-aware layouts with collective operations at multiple granularities.

## Pre-Binary Validation

A 1,024-thread `fx.coop.BlockScan[fx.Int32, 1024]` was compiled for `gfx942` through FlyDSL's pipeline up to `reconcile-unrealized-casts`. No GPU execution or binary generation was used.

The current implementation linearly folds preceding wave totals in every thread. Its lowered kernel contained:

- 15 static LDS-load sites for cross-wave prefixes;
- 23 `llvm.add` operations;
- 16 `llvm.select` operations;
- approximately 7.4 KB of LLVM-dialect kernel IR.

A temporary hierarchical prototype instead had wave 0 scan the 16 wave totals, write the prefixes back, and let each thread read only the preceding wave's prefix. Its lowered kernel contained:

- 2 static LDS-load sites for the hierarchy;
- 15 `llvm.add` operations;
- 1 `llvm.select` operation;
- approximately 4.9 KB of LLVM-dialect kernel IR;
- one additional block barrier.

This is not a runtime-performance claim. It demonstrates that the cooperative layer contains measurable optimization opportunities that can be developed and reviewed using pre-binary IR.

## Suggested Pull-Request Sequence

1. **Optimize the existing BlockScan hierarchy.** Replace the per-thread linear cross-wave fold with a single-wave scan of the wave totals. Add pre-binary checks for both wave64 (`gfx942`) and wave32 (`gfx1201`).
2. **Add portable `BlockExchange`.** Support blocked-to-striped, striped-to-blocked, and blocked-to-warp-striped using padded LDS. Add exhaustive host-side permutation tests.
3. **Add wave-local lowering.** Recognize conversions whose ownership remains inside one wave and emit packed DPP or `ds_bpermute` paths. Cover 8-, 16-, 32-, and 64-bit values and partial logical waves.
4. **Generalize around Fly layouts.** Extract a static redistribution planner after the named exchange API has established its semantics and tests.

The first maintainer discussion should propose steps 1 and 2 under issue #1016, with layout-directed lowering described as the longer-term design.

## Focused FlyDSL Reading Path

The following order builds the necessary context without requiring a tour of the entire repository.

### 1. Understand Fly layout semantics

- `docs/layout_system_guide.md` — user-facing layout algebra and terminology.
- `docs/cute_layout_algebra_guide.md` — the mathematical background and relation to CuTe layouts.
- `include/flydsl/Dialect/Fly/IR/FlyAttrDefs.td` — compile-time representation of layouts, tuples, tiles, and swizzles.
- `include/flydsl/Dialect/Fly/IR/FlyTypeDefs.td` — where layouts are attached to `MemRef`, `CoordTensor`, `TiledCopy`, and `TiledMma` types.
- `lib/Dialect/Fly/Utils/NormalForm.cpp` — normalization machinery used by layout algebra.

Questions to keep in mind: how does a nested `(thread, value)` coordinate map to a logical element, and under what conditions is that mapping invertible?

### 2. Follow layout construction from Python into MLIR

- `python/flydsl/expr/primitive.py` — Python APIs such as `make_layout`, composition, divide, partition, `copy`, and `gemm`.
- `python/flydsl/expr/typing.py` — especially `Layout`, `Tensor`, and `Vector`. Notice that `Vector` has a local shape and dtype but no block-wide distribution layout.
- `python/flydsl/expr/derived.py` — Python wrappers for copy/MMA atoms and their thread slices.
- `python/flydsl/_mlir/dialects/fly.py` or the generated package equivalent — Python bindings used by the expression layer.

### 3. Understand thread/value partitioning and register promotion

- `include/flydsl/Dialect/Fly/Utils/TiledOpUtils.h` — construction of tiled MMA/copy layouts and partitions.
- `lib/Dialect/Fly/Transforms/LayoutLowering.cpp` — the main lowering of layout algebra, including vectorized load/store permutation helpers.
- `lib/Dialect/Fly/Transforms/ConvertAtomCallToSSAForm.cpp` — converts register-memory atom calls to SSA form.
- `lib/Dialect/Fly/Transforms/PromoteRegMemToVectorSSA.cpp` — turns register-memory tensors into vector SSA values.
- `tests/mlir/Transforms/promote_regmem_to_ssa.mlir` — compact examples of the before/after representation.

The key boundary for this proposal is where a richly laid-out register memref becomes a plain vector SSA value.

### 4. Study the new cooperative library

- `python/flydsl/extension/coop/_common.py` — identities, combining operations, thread indexing, and per-thread partials.
- `python/flydsl/extension/coop/warp/scan.py` — portable ordered wave scan.
- `python/flydsl/extension/coop/warp/rocdl.py` — AMD-specific DPP and lane-operation implementations.
- `python/flydsl/extension/coop/block/scan.py` — current block scan, its algorithm registry, shared storage, and the linear cross-wave TODO.
- `python/flydsl/extension/coop/block/reduce.py` — useful comparison of the warp-reduction and raking policies.
- `python/flydsl/extension/coop/block/_spec.py` — specialization by dtype, block shape, algorithm, and target wave size.

Then read the tests:

- `tests/extension/coop/test_warp_scan.py`
- `tests/extension/coop/test_warp_rocdl.py`
- `tests/extension/coop/test_block_scan.py`
- `tests/extension/coop/test_block_reduce.py`

These establish conventions that a new `BlockExchange` should follow.

### 5. Inspect existing manual redistributions

- `kernels/attention/fused_rope_cache_kernel.py` — cross-lane pairing with packed `ds_bpermute`.
- `kernels/moe/moe_sorting_kernel.py` — DPP plus `ds_bpermute` prefix operations and cross-wave scratch storage.
- `kernels/mega_moe/dispatch.py` — another independent cross-lane exchange implementation.
- `kernels/attention/pa_decode_swa.py` — lane-to-lane weight movement.
- `kernels/moe/moe_gemm_2stage/gemm1.py` and `gemm2.py` — LDS C-shuffle epilogues.

These are candidate consumers and a useful catalogue of the cases a planner would eventually need to express.

### 6. Understand target lowering and pre-binary testing

- `python/flydsl/compiler/backends/rocm.py` — the exact pre-binary pass pipeline and wave32/wave64 target configuration.
- `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp` — lowering from Fly operations to upstream MLIR and ROCDL operations.
- `python/flydsl/expr/rocdl/__init__.py` — exposed AMD lane, DPP, buffer, and LDS primitives.
- `tests/mlir/Conversion/gpu_ops.mlir` — examples of FileCheck-based ROCDL lowering tests.
- `tests/unit/test_external_llvm_codegen.py` — demonstrates running the pipeline only through its pre-binary fragments.

For the first implementation, tests should inspect MLIR/LLVM dialect output rather than attempting to assert final ISA.

## External Code to Compare

- `/home/qiping-pan/Documents/workspace/triton/lib/Analysis/Utility.cpp` — `ScanLoweringHelper` and layout analysis.
- `/home/qiping-pan/Documents/workspace/triton/lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp` — hierarchical scan lowering.
- `/home/qiping-pan/Documents/workspace/triton/include/triton/Tools/LinearLayout.h` — register/lane/warp/block layout model.
- `/home/qiping-pan/Documents/workspace/triton/lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp` — conversion routing and lowering.
- `/home/qiping-pan/Documents/workspace/tilelang/tilelang/language/scan_op.py` — TileLang's scan surface and shared-memory staging.
- `/home/qiping-pan/Documents/workspace/composable_kernel/include/ck_tile/core/` — CK Tile's compile-time tensor/layout utilities.
