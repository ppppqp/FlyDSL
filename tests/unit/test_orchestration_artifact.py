# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Public compiler-artifact boundary for external orchestration layers."""

import pickle

import pytest

from flydsl._mlir import ir
from flydsl.compiler.jit_executor import (
    CompiledArtifact,
    KernelLaunch,
    LaunchPlan,
    LaunchPlanError,
    extract_launch_plan,
)
from flydsl.compiler.jit_function import _create_mlir_context

pytestmark = [pytest.mark.l0_backend_agnostic]


class _Module:
    def __str__(self):
        return 'module { gpu.binary @kernels [#gpu.object<"...">] }'


def _artifact():
    return CompiledArtifact(
        _Module(),
        "launch",
        "module { gpu.launch_func @kernels::@kernel }",
        backend="rocm",
        target="gfx942",
        kernel_abi="rocm.bare_ptr",
    )


def test_orchestration_export_is_public_and_runtime_independent():
    exported = _artifact().export_for_orchestration()

    assert exported.host_entry == "launch"
    assert exported.backend == "rocm"
    assert exported.target == "gfx942"
    assert exported.kernel_abi == "rocm.bare_ptr"
    assert "gpu.launch_func" in exported.source_ir
    assert "gpu.binary" in exported.compiled_ir
    assert exported.device_objects == ()


def test_orchestration_metadata_survives_disk_cache_round_trip():
    restored = pickle.loads(pickle.dumps(_artifact()))
    exported = restored.export_for_orchestration()

    assert exported.backend == "rocm"
    assert exported.target == "gfx942"
    assert exported.kernel_abi == "rocm.bare_ptr"
    assert "gpu.launch_func" in exported.source_ir
    assert exported.device_objects == ()


def test_launch_plan_survives_disk_cache_round_trip():
    artifact = CompiledArtifact(
        _Module(),
        "launch",
        "module { gpu.launch_func @kernels::@kernel }",
        backend="rocm",
        target="gfx942",
        kernel_abi="rocm.bare_ptr",
        launch_plan=LaunchPlan(
            host_entry="launch",
            launches=(KernelLaunch(0, "@kernels::@kernel", (1, 1, 1), (64, 1, 1), 0, (), ()),),
        ),
    )

    exported = pickle.loads(pickle.dumps(artifact)).export_for_orchestration()
    assert exported.launch_plan is not None
    assert exported.launch_plan.launches[0].block == (64, 1, 1)


def test_device_objects_are_copied_out_of_the_producing_mlir_runtime():
    context = _create_mlir_context()
    with context:
        module = ir.Module.parse("""module {
              gpu.binary @kernels [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "\\7FELF">]
            }""")
        artifact = CompiledArtifact(
            module,
            "launch",
            "module { func.func @launch() { return } }",
            backend="rocm",
            target="gfx942",
            kernel_abi="rocm.bare_ptr",
        )

    exported = artifact.export_for_orchestration()
    assert len(exported.device_objects) == 1
    assert exported.device_objects[0].data == b"\x7fELF"
    assert "gfx942" in exported.device_objects[0].target
    restored = pickle.loads(pickle.dumps(artifact)).export_for_orchestration()
    assert restored.device_objects == exported.device_objects


def test_legacy_artifact_requires_recompilation():
    artifact = CompiledArtifact(_Module(), "launch")

    with pytest.raises(RuntimeError, match="recompile"):
        artifact.export_for_orchestration()


def _parse_launch_module(body):
    return ir.Module.parse(f"""module attributes {{gpu.container_module}} {{
          gpu.module @kernels {{
            gpu.func @stage(%arg0: !fly.ptr<f32, global>, %arg1: i32) kernel {{ gpu.return }}
          }}
          func.func @launch(%arg0: !fly.ptr<f32, global>, %arg1: i32) {{
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            {body}
            return
          }}
        }}""")


def test_extracts_restricted_straight_line_launch_plan():
    context = _create_mlir_context()
    with context:
        module = _parse_launch_module("""gpu.launch_func @kernels::@stage
                 blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)
                 args(%arg0 : !fly.ptr<f32, global>, %arg1 : i32)
               gpu.launch_func @kernels::@stage
                 blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)
                 args(%arg0 : !fly.ptr<f32, global>, %arg1 : i32)""")
        plan = extract_launch_plan(module, "launch", ("output", "count"))

    assert len(plan.launches) == 2
    assert plan.launches[0].kernel == "@kernels::@stage"
    assert plan.launches[0].block == (64, 1, 1)
    assert plan.launches[0].arguments[0].kind == "resource"
    assert plan.launches[0].arguments[0].binding == "output"
    assert plan.launches[0].arguments[1].kind == "scalar"
    assert plan.launches[1].dependencies == (0,)


def test_direct_host_memref_is_preserved_for_physical_abi_expansion():
    context = _create_mlir_context()
    with context:
        module = ir.Module.parse("""module attributes {gpu.container_module} {
          gpu.module @kernels {
            gpu.func @stage(%arg0: !fly.memref<f32, global, (?):(1)>) kernel { gpu.return }
          }
          func.func @launch(%arg0: !fly.memref<f32, global, (?):(1)>) {
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            gpu.launch_func @kernels::@stage
              blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)
              args(%arg0 : !fly.memref<f32, global, (?):(1)>)
            return
          }
        }""")
        plan = extract_launch_plan(module, "launch", ("output",))

    argument = plan.launches[0].arguments[0]
    assert argument.kind == "memref"
    assert argument.binding == "output"


def test_rejects_launch_nested_in_control_flow():
    context = _create_mlir_context()
    with context:
        module = _parse_launch_module("""%condition = arith.constant true
               scf.if %condition {
                 gpu.launch_func @kernels::@stage
                   blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)
                   args(%arg0 : !fly.ptr<f32, global>, %arg1 : i32)
               }""")
        with pytest.raises(LaunchPlanError, match="control flow"):
            extract_launch_plan(module, "launch", ("output", "count"))


def test_rejects_computed_kernel_argument():
    context = _create_mlir_context()
    with context:
        module = _parse_launch_module("""%computed = arith.addi %arg1, %arg1 : i32
               gpu.launch_func @kernels::@stage
                 blocks in (%c1, %c1, %c1) threads in (%c64, %c1, %c1)
                 args(%arg0 : !fly.ptr<f32, global>, %computed : i32)""")
        with pytest.raises(LaunchPlanError, match="view-derived"):
            extract_launch_plan(module, "launch", ("output", "count"))
