# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Public compiler-artifact boundary for external orchestration layers."""

import pickle

import pytest

from flydsl._mlir import ir
from flydsl.compiler.jit_executor import CompiledArtifact
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


def test_legacy_artifact_requires_recompilation():
    artifact = CompiledArtifact(_Module(), "launch")

    with pytest.raises(RuntimeError, match="recompile"):
        artifact.export_for_orchestration()
