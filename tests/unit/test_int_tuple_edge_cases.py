# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, fly
from flydsl._mlir.passmanager import PassManager


@pytest.mark.parametrize("value", [1.5, "bad", object()])
def test_native_invalid_leaf_reports_type(value):
    with ir.Context(), ir.Location.unknown():
        with pytest.raises(ValueError, match=type(value).__name__):
            fly.IntTupleType.get(value)


@pytest.mark.parametrize("dtype", [fx.Float16, fx.BFloat16, fx.Float32, fx.Float64])
@pytest.mark.parametrize("dynamic", [False, True])
def test_float_wrappers_are_rejected(dtype, dynamic):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            value = dtype(1.5)
            if dynamic:
                raw = arith.ConstantOp(dtype.ir_type, ir.FloatAttr.get(dtype.ir_type, 1.5)).result
                value = dtype(raw)
            with pytest.raises(TypeError, match="must be integers"):
                fx.make_int_tuple(value)
            with pytest.raises(TypeError, match="must be integers"):
                fx.make_stride(fx.Basis(value, [0]))


@pytest.mark.parametrize("value", [2**31, -(2**31) - 1, 2**32, 2**63, 2**100, -(2**100)])
@pytest.mark.parametrize("basis", [False, True])
def test_static_integer_overflow_is_rejected(value, basis):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            leaf = fx.Basis(value, [0]) if basis else value
            with pytest.raises(OverflowError):
                fx.make_int_tuple((1, leaf))


@pytest.mark.parametrize("value", [-(2**31), -1, 0, 2**31 - 1])
def test_static_integer_boundaries_roundtrip(value):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            assert fx.make_int_tuple(value).to_py_value() == value
            assert fx.make_stride(fx.Basis(value, [0])).to_py_value() == (fx.Basis(value, [0]),)
        assert module.operation.verify()


@pytest.mark.parametrize("value", [None, (2, None), (None, (3, None)), (), (2, ()), ((), ())])
def test_static_structure_roundtrip_and_ir_parse(value):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            result = fx.make_int_tuple(value)
            assert result.depth >= 0
            assert result.to_py_value() == value
            assert fx.make_int_tuple(result).type == result.type
        assert module.operation.verify()
        parsed = ir.Module.parse(str(module))
        assert parsed.operation.verify()
        PassManager.parse("builtin.module(canonicalize,cse)").run(parsed.operation)
        assert parsed.operation.verify()


def test_dynamic_tuple_with_empty_modes_roundtrips():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            raw = arith.ConstantOp(fx.Int32.ir_type, ir.IntegerAttr.get(fx.Int32.ir_type, 7)).result
            result = fx.make_int_tuple(((), fx.Int32(raw), ()))
            assert result.depth == 2
            rebuilt = fx.make_int_tuple(result.to_py_value())
            assert rebuilt.type == result.type
        assert module.operation.verify()
        PassManager.parse("builtin.module(fly-canonicalize,fly-layout-lowering,canonicalize)").run(module.operation)
        assert module.operation.verify()
        parsed = ir.Module.parse(str(module))
        assert parsed.operation.verify()
        PassManager.parse("builtin.module(canonicalize,cse)").run(parsed.operation)
        assert parsed.operation.verify()


def test_dynamic_coord_preserves_wildcards_on_rebuild():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            raw = arith.ConstantOp(fx.Int32.ir_type, ir.IntegerAttr.get(fx.Int32.ir_type, 7)).result
            coord = fx.make_coord(None, (fx.Int32(raw), None))
            rebuilt = fx.make_coord(*coord.to_py_value())
            assert rebuilt.type == coord.type
            assert rebuilt.to_py_value()[0] is None
            assert rebuilt.to_py_value()[1][1] is None
        assert module.operation.verify()
