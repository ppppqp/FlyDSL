#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compare the current RoPE kernel with a git revision using GPU graphs.

Run from the repository root with the built package and root on PYTHONPATH::

    HIP_VISIBLE_DEVICES=0 ARCH=gfx1200 PYTHONPATH=build-fly/python_packages:. \
        .venv/bin/python scripts/benchmark_rope_coop.py --baseline-ref COMMIT --output results.json

Each variant gets its own output buffers. Kernels are compiled and checked
before timing; alternating measurements use pre-captured graphs so Python
launch overhead is excluded. The reported median and range describe repeated
measurements on this device, not a cross-architecture performance guarantee.
"""

import argparse
import importlib.util
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import torch

from kernels.attention.fused_rope_cache_kernel import build_fused_rope_cache_module


def load_baseline(revision, directory):
    path = "kernels/attention/fused_rope_cache_kernel.py"
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
    baseline_path = Path(directory) / "baseline_rope.py"
    baseline_path.write_text(source)
    spec = importlib.util.spec_from_file_location("baseline_rope", baseline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_fused_rope_cache_module


def make_case(builder, tensors, tokens, heads, dim, dtype_str, flash):
    q, k, v, positions, cos, sin, slots = tensors
    blocks = max(4, (tokens + 15) // 16)
    if flash:
        kc_shape = vc_shape = (blocks, 16, 1, dim)
    else:
        kc_shape, vc_shape = (blocks, 1, dim // 16, 16, 16), (blocks, 1, dim, 16)
    kc = torch.zeros(kc_shape, device="cuda", dtype=q.dtype)
    vc = torch.zeros(vc_shape, device="cuda", dtype=q.dtype)
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    launch = builder(head_dim=dim, num_q_heads=heads, num_kv_heads=1, flash_layout=flash, dtype_str=dtype_str)

    def call():
        launch(
            q,
            k,
            v,
            positions,
            cos,
            sin,
            slots,
            kc,
            vc,
            qo,
            ko,
            tokens,
            scale,
            scale,
            stream=torch.cuda.current_stream(),
        )

    call()
    torch.cuda.synchronize()
    return call, (qo, ko, kc, vc)


def capture(call, launches):
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(5):
            call()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            call()
    return graph


def measure(graph, launches, replays):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (launches * replays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--warmup-replays", type=int, default=256)
    parser.add_argument("--launches", type=int, default=64)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    if min(args.rounds, args.launches, args.replays, args.warmup_replays) < 1:
        parser.error("rounds, launches, replays, and warmup-replays must be positive")
    baseline_ref = subprocess.check_output(["git", "rev-parse", args.baseline_ref], text=True).strip()
    properties = torch.cuda.get_device_properties(0)
    report = dict(
        baseline_ref=baseline_ref,
        gpu=properties.name,
        arch=properties.gcnArchName,
        torch=torch.__version__,
        rounds=args.rounds,
        warmup_replays=args.warmup_replays,
        launches=args.launches,
        replays=args.replays,
        timing="HIP events over captured graphs; alternating variant order",
        cases=[],
    )
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="flydsl-rope-baseline-") as directory:
        baseline = load_baseline(baseline_ref, directory)
        for dtype_str, dtype in (("bf16", torch.bfloat16), ("f16", torch.float16)):
            for dim in (32, 64, 128, 256):
                for tokens in (1, 32, 128):
                    for flash in (True, False):
                        heads = 8
                        q = torch.randn(tokens, heads, dim, device="cuda", dtype=dtype)
                        k, v = [torch.randn(tokens, 1, dim, device="cuda", dtype=dtype) for _ in range(2)]
                        positions = torch.randint(0, 1024, (tokens,), device="cuda", dtype=torch.int32)
                        cos, sin = [torch.randn(1024, dim // 2, device="cuda", dtype=dtype) for _ in range(2)]
                        slots = torch.arange(tokens, device="cuda", dtype=torch.int32)
                        tensors = (q, k, v, positions, cos, sin, slots)
                        old, old_out = make_case(baseline, tensors, tokens, heads, dim, dtype_str, flash)
                        new, new_out = make_case(
                            build_fused_rope_cache_module, tensors, tokens, heads, dim, dtype_str, flash
                        )
                        for a, b in zip(old_out, new_out):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        graphs = [capture(call, args.launches) for call in (old, new)]
                        # Compilation/capture can leave a short kernel at idle
                        # clocks. Sustain GPU work on both variants before the
                        # alternating measurements, rather than timing ramp-up.
                        for _ in range(args.warmup_replays):
                            for graph in graphs:
                                graph.replay()
                        torch.cuda.synchronize()
                        timings = [[], []]
                        for round_id in range(args.rounds):
                            for variant in ((0, 1) if round_id % 2 == 0 else (1, 0)):
                                timings[variant].append(measure(graphs[variant], args.launches, args.replays))
                        medians = [statistics.median(samples) for samples in timings]
                        case = dict(
                            dtype=dtype_str,
                            head_dim=dim,
                            tokens=tokens,
                            q_heads=heads,
                            kv_heads=1,
                            flash_layout=flash,
                            baseline_us=medians[0],
                            cooperative_us=medians[1],
                            speedup=medians[0] / medians[1],
                            samples_us=timings,
                        )
                        report["cases"].append(case)
                        print(
                            f"{dtype_str} D={dim} T={tokens} flash={flash}: "
                            f"{medians[0]:.3f} -> {medians[1]:.3f} us ({case['speedup']:.3f}x)",
                            flush=True,
                        )
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
