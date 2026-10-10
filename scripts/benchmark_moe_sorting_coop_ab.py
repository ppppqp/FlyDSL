#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A/B benchmark the manual-collective and ``fx.coop`` MoE sorting kernels.

The two implementations are ordinary Python modules in the same checkout:

* ``kernels.moe.moe_sorting_kernel_baseline`` is the main-branch baseline.
* ``kernels.moe.moe_sorting_kernel`` is the ``feat/coop-reduce-scan`` candidate.

Both variants receive identical inputs and separately allocated outputs. The
correctness gate validates the complete routing result semantically, checks
whether the two physical output layouts are also identical, and verifies that
``moe_buf`` was zeroed. Timing starts only after both variants have compiled
and warmed up. Candidate-first and baseline-first samples alternate to reduce
order bias.

Example::

    PYTHONPATH=. python scripts/benchmark_moe_sorting_coop_ab.py

    PYTHONPATH=. python scripts/benchmark_moe_sorting_coop_ab.py \
        --tokens 1,16,17,128,512,2048,4096,8192 \
        --experts 256 --topk 8 --model-dim 7168 \
        --warmup 10 --samples 50 --calls-per-sample 10 \
        --json /workspace/results/moe-sorting-coop-ab.json

The installed FlyDSL wheel supplies the compiler and embedded MLIR runtime;
the two kernel modules above come from this source checkout. Runtime disk
caching is disabled deliberately so the same-named kernels in the two modules
cannot alias through an old artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# This must be set before importing FlyDSL or either kernel module.
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import torch

import flydsl
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch
from kernels.moe import moe_sorting_kernel as candidate
from kernels.moe import moe_sorting_kernel_baseline as baseline

BASELINE_REVISION = "b81e99d69fcb2cd6100502b6360d643be898d669"
CANDIDATE_REVISION = "3b8501176ed5202eeb7a64b5b57232d4894a634e"
UNIT_SIZE = 32


@dataclass
class Outputs:
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    moe_buf: torch.Tensor
    workspace: torch.Tensor | None


@dataclass
class Snapshot:
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    moe_buf_is_zero: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_tokens(text: str) -> list[int]:
    tokens = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not tokens or any(value <= 0 for value in tokens):
        raise argparse.ArgumentTypeError("--tokens must contain positive comma-separated integers")
    return tokens


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "p10_us": _percentile(values, 0.10),
        "p90_us": _percentile(values, 0.90),
        "min_us": min(values),
        "mean_us": statistics.fmean(values),
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _path_for(module: Any, tokens: int, experts: int, topk: int) -> str:
    sub_tokens = module._compute_sub_tokens(experts)
    oneshot_max = min(sub_tokens, max(16, module.BLOCK_SIZE // max(topk, experts // 8)))
    if tokens <= min(sub_tokens, oneshot_max):
        return "oneshot"
    return "p0v2+p23" if tokens <= 2048 else "k1+k2+p1+p23"


def _make_inputs(tokens: int, experts: int, topk: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # topk() over random scores gives unique experts for each token, matching
    # the router constraint without a Python loop over tokens. Generate these
    # untimed inputs on the CPU so the benchmark does not depend on rocRAND or
    # PyTorch's GPU top-k kernels; only the FlyDSL kernels are under test.
    scores = torch.rand((tokens, experts), generator=generator)
    topk_ids = scores.topk(topk, dim=1, sorted=False).indices.to(torch.int32).contiguous().cuda()
    topk_weights = torch.rand((tokens, topk), dtype=torch.float32, generator=generator).cuda()
    return topk_ids, topk_weights


def _allocate(module: Any, tokens: int, experts: int, topk: int, model_dim: int) -> Outputs:
    max_padded = tokens * topk + experts * UNIT_SIZE - topk
    max_blocks = (max_padded + UNIT_SIZE - 1) // UNIT_SIZE
    workspace_elements = module.moe_sorting_get_workspace_size(tokens, experts, topk, UNIT_SIZE)
    workspace = torch.empty(workspace_elements, dtype=torch.int32, device="cuda") if workspace_elements else None
    return Outputs(
        sorted_ids=torch.empty(max_padded, dtype=torch.int32, device="cuda"),
        sorted_weights=torch.empty(max_padded, dtype=torch.float32, device="cuda"),
        sorted_expert_ids=torch.empty(max_blocks, dtype=torch.int32, device="cuda"),
        num_valid_ids=torch.empty(2, dtype=torch.int32, device="cuda"),
        moe_buf=torch.empty((tokens, model_dim), dtype=torch.bfloat16, device="cuda"),
        workspace=workspace,
    )


def _reset(outputs: Outputs, sentinel: int) -> None:
    outputs.sorted_ids.fill_(sentinel)
    outputs.sorted_weights.fill_(float("nan"))
    outputs.sorted_expert_ids.fill_(-1)
    outputs.num_valid_ids.fill_(-1)
    outputs.moe_buf.fill_(1)


def _call(module: Any, topk_ids: torch.Tensor, topk_weights: torch.Tensor, outputs: Outputs, experts: int) -> None:
    module.moe_sorting_flydsl(
        topk_ids,
        topk_weights,
        outputs.sorted_ids,
        outputs.sorted_weights,
        outputs.sorted_expert_ids,
        outputs.num_valid_ids,
        outputs.moe_buf,
        experts,
        UNIT_SIZE,
        workspace=outputs.workspace,
    )


def _snapshot(outputs: Outputs) -> Snapshot:
    torch.cuda.synchronize()
    return Snapshot(
        sorted_ids=outputs.sorted_ids.cpu(),
        sorted_weights=outputs.sorted_weights.cpu(),
        sorted_expert_ids=outputs.sorted_expert_ids.cpu(),
        num_valid_ids=outputs.num_valid_ids.cpu(),
        moe_buf_is_zero=bool(torch.count_nonzero(outputs.moe_buf).item() == 0),
    )


def _validate_snapshot(
    name: str,
    snapshot: Snapshot,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
) -> tuple[bool, dict[str, Any]]:
    ids = topk_ids.cpu().to(torch.int64)
    weights = topk_weights.cpu()
    tokens, topk = ids.shape
    sentinel = (topk << 24) | tokens

    counts = torch.bincount(ids.reshape(-1), minlength=experts)
    blocks_per_expert = (counts + UNIT_SIZE - 1) // UNIT_SIZE
    expected_eids = torch.repeat_interleave(torch.arange(experts, dtype=torch.int32), blocks_per_expert)
    expected_padded = int(blocks_per_expert.sum().item()) * UNIT_SIZE
    expected_valid = tokens * topk

    errors: list[str] = []
    got_nv = snapshot.num_valid_ids.tolist()
    if got_nv != [expected_padded, tokens]:
        errors.append(f"num_valid_ids={got_nv}, expected={[expected_padded, tokens]}")

    got_eids = snapshot.sorted_expert_ids[: len(expected_eids)]
    if not torch.equal(got_eids, expected_eids):
        errors.append("sorted_expert_ids differ from expected expert-block sequence")

    padded_ids = snapshot.sorted_ids[:expected_padded].to(torch.int64)
    valid_mask = padded_ids != sentinel
    if int(valid_mask.sum().item()) != expected_valid:
        errors.append(f"valid packed IDs={int(valid_mask.sum())}, expected={expected_valid}")

    packed = padded_ids[valid_mask]
    packed_weights = snapshot.sorted_weights[:expected_padded][valid_mask]
    if packed.numel():
        token_ids = packed & 0xFFFFFF
        slots = packed >> 24
        bounds_ok = bool(((token_ids >= 0) & (token_ids < tokens) & (slots >= 0) & (slots < topk)).all())
        if not bounds_ok:
            errors.append("one or more packed IDs decode outside the input bounds")
        else:
            actual_experts = ids[token_ids, slots].to(torch.int32)
            position_experts = got_eids[torch.arange(expected_padded)[valid_mask] // UNIT_SIZE]
            if not torch.equal(actual_experts, position_experts):
                errors.append("one or more packed IDs were placed in the wrong expert block")

            expected_weights = weights[token_ids, slots]
            if not torch.equal(packed_weights, expected_weights):
                max_error = float((packed_weights - expected_weights).abs().max().item())
                errors.append(f"sorted_weights differ from their routed inputs (max_abs={max_error})")

            expected_packed = (
                (torch.arange(topk, dtype=torch.int64).view(1, -1) << 24)
                | torch.arange(tokens, dtype=torch.int64).view(-1, 1)
            ).reshape(-1)
            if not torch.equal(torch.sort(packed).values, torch.sort(expected_packed).values):
                errors.append("packed-ID multiset does not contain every input route exactly once")

    if not snapshot.moe_buf_is_zero:
        errors.append("moe_buf was not completely zeroed")

    ok = not errors
    detail = {
        "ok": ok,
        "expected_padded": expected_padded,
        "expected_valid": expected_valid,
        "valid_blocks": len(expected_eids),
        "errors": errors,
    }
    status = "PASS" if ok else "FAIL"
    print(f"    {name:<10} {status}: padded={expected_padded}, valid={expected_valid}, blocks={len(expected_eids)}")
    for error in errors:
        print(f"      - {error}")
    return ok, detail


def _compare_layouts(baseline_snapshot: Snapshot, candidate_snapshot: Snapshot, sentinel: int) -> dict[str, bool]:
    num_padded = int(baseline_snapshot.num_valid_ids[0].item())
    num_blocks = num_padded // UNIT_SIZE
    ids_equal = torch.equal(baseline_snapshot.sorted_ids[:num_padded], candidate_snapshot.sorted_ids[:num_padded])
    eids_equal = torch.equal(
        baseline_snapshot.sorted_expert_ids[:num_blocks], candidate_snapshot.sorted_expert_ids[:num_blocks]
    )
    valid = baseline_snapshot.sorted_ids[:num_padded] != sentinel
    # Use each variant's packed IDs to select meaningful weight positions. In
    # normal operation the ID layouts are equal; semantic validation above is
    # authoritative if a legal ordering difference appears.
    weights_equal = ids_equal and torch.equal(
        baseline_snapshot.sorted_weights[:num_padded][valid],
        candidate_snapshot.sorted_weights[:num_padded][valid],
    )
    return {"ids": ids_equal, "weights": weights_equal, "expert_ids": eids_equal}


def _time_interleaved(
    baseline_fn,
    candidate_fn,
    *,
    warmup: int,
    samples: int,
    calls_per_sample: int,
    flush_mb: int,
) -> tuple[dict[str, float], dict[str, float]]:
    for _ in range(warmup):
        baseline_fn()
        candidate_fn()
    torch.cuda.synchronize()

    flush = torch.empty(flush_mb * 1024 * 1024, dtype=torch.uint8, device="cuda") if flush_mb else None
    timings: dict[str, list[float]] = {"baseline": [], "candidate": []}
    functions = {"baseline": baseline_fn, "candidate": candidate_fn}

    for sample in range(samples):
        order = ("baseline", "candidate") if sample % 2 == 0 else ("candidate", "baseline")
        for name in order:
            if flush is not None:
                flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(calls_per_sample):
                functions[name]()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000.0 / calls_per_sample)

    return _stats(timings["baseline"]), _stats(timings["candidate"])


def _run_case(args, tokens: int, seed: int) -> tuple[bool, dict[str, Any]]:
    experts, topk, model_dim = args.experts, args.topk, args.model_dim
    topk_ids, topk_weights = _make_inputs(tokens, experts, topk, seed)
    base_outputs = _allocate(baseline, tokens, experts, topk, model_dim)
    cand_outputs = _allocate(candidate, tokens, experts, topk, model_dim)
    sentinel = (topk << 24) | tokens

    base_path = _path_for(baseline, tokens, experts, topk)
    cand_path = _path_for(candidate, tokens, experts, topk)
    if base_path != cand_path:
        raise RuntimeError(f"variant path mismatch: baseline={base_path}, candidate={cand_path}")

    print(f"\n  T={tokens}, E={experts}, topk={topk}, model_dim={model_dim}, path={base_path}")

    _reset(base_outputs, sentinel)
    _reset(cand_outputs, sentinel)
    _call(baseline, topk_ids, topk_weights, base_outputs, experts)
    _call(candidate, topk_ids, topk_weights, cand_outputs, experts)
    base_snapshot = _snapshot(base_outputs)
    cand_snapshot = _snapshot(cand_outputs)

    base_ok, base_correctness = _validate_snapshot("baseline", base_snapshot, topk_ids, topk_weights, experts)
    cand_ok, cand_correctness = _validate_snapshot("candidate", cand_snapshot, topk_ids, topk_weights, experts)
    layout = _compare_layouts(base_snapshot, cand_snapshot, sentinel)
    exact_layout = all(layout.values())
    print(
        "    physical  "
        + ("IDENTICAL" if exact_layout else "DIFFERS")
        + f": ids={layout['ids']} weights={layout['weights']} expert_ids={layout['expert_ids']}"
    )

    def baseline_fn():
        _call(baseline, topk_ids, topk_weights, base_outputs, experts)

    def candidate_fn():
        _call(candidate, topk_ids, topk_weights, cand_outputs, experts)

    base_stats, cand_stats = _time_interleaved(
        baseline_fn,
        candidate_fn,
        warmup=args.warmup,
        samples=args.samples,
        calls_per_sample=args.calls_per_sample,
        flush_mb=args.flush_mb,
    )
    ratio = cand_stats["median_us"] / base_stats["median_us"]
    regression_pct = (ratio - 1.0) * 100.0
    perf_ok = regression_pct <= args.max_regression_pct
    print(
        f"    baseline  median={base_stats['median_us']:.3f} us "
        f"p10={base_stats['p10_us']:.3f} p90={base_stats['p90_us']:.3f}"
    )
    print(
        f"    candidate median={cand_stats['median_us']:.3f} us "
        f"p10={cand_stats['p10_us']:.3f} p90={cand_stats['p90_us']:.3f}"
    )
    print(f"    ratio={ratio:.4f}x regression={regression_pct:+.2f}% ({'PASS' if perf_ok else 'FAIL'})")

    correctness_ok = base_ok and cand_ok and (exact_layout or not args.require_exact_layout)
    passed = correctness_ok and perf_ok
    return passed, {
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "model_dim": model_dim,
        "path": base_path,
        "baseline_correctness": base_correctness,
        "candidate_correctness": cand_correctness,
        "physical_layout_equal": layout,
        "baseline": base_stats,
        "candidate": cand_stats,
        "candidate_over_baseline": ratio,
        "regression_pct": regression_pct,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_parse_tokens, default=_parse_tokens("1,16,17,128,512,2048,4096,8192"))
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-sample", type=int, default=10)
    parser.add_argument("--flush-mb", type=int, default=0, help="optional cache-flush allocation per timed sample")
    parser.add_argument("--max-regression-pct", type=float, default=5.0)
    parser.add_argument("--require-exact-layout", action="store_true")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA-visible ROCm GPU is required")
    if args.topk <= 0 or args.topk > args.experts:
        parser.error("--topk must be in [1, experts]")
    for name in ("experts", "model_dim", "warmup", "samples", "calls_per_sample"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.flush_mb < 0:
        parser.error("--flush-mb cannot be negative")
    if not hasattr(fx.coop, "BlockScan") or not hasattr(fx.coop, "warp_scan_with_aggregate"):
        parser.error("the installed FlyDSL wheel does not provide the cooperative APIs required by the candidate")

    arch = str(get_rocm_arch() or "unknown")
    if arch != "gfx942":
        print(f"WARNING: designed for MI300X/gfx942, detected {arch}", file=sys.stderr)

    root = Path(__file__).resolve().parents[1]
    baseline_file = root / "kernels/moe/moe_sorting_kernel_baseline.py"
    candidate_file = root / "kernels/moe/moe_sorting_kernel.py"
    metadata = {
        "device": torch.cuda.get_device_name(0),
        "arch": arch,
        "torch": str(torch.__version__),
        "torch_hip": str(torch.version.hip),
        "flydsl": str(getattr(flydsl, "__version__", "unknown")),
        "baseline_revision": BASELINE_REVISION,
        "candidate_revision": CANDIDATE_REVISION,
        "baseline_sha256": _sha256(baseline_file),
        "candidate_sha256": _sha256(candidate_file),
        "warmup": args.warmup,
        "samples": args.samples,
        "calls_per_sample": args.calls_per_sample,
        "flush_mb": args.flush_mb,
        "max_regression_pct": args.max_regression_pct,
    }
    print("MoE sorting cooperative-API A/B")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    rows: list[dict[str, Any]] = []
    all_passed = True
    for index, tokens in enumerate(args.tokens):
        passed, row = _run_case(args, tokens, args.seed + index)
        rows.append(row)
        all_passed &= passed

    payload = {"metadata": metadata, "results": rows, "passed": all_passed}
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    print(f"\noverall: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
