// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

/*
Why do we need this pass?

A type such as !fly.layout<(16, ?):(1, ?)> is meaning to the Fly compiler, but LLVM do not know how
to pass a "fly layout". Before reaching LLVM, the compiler must answer "what concrete runtime bits
represent this value?"
For this layout, the answer is just the two dynamic integers. The static 16 and 1 already exist in
the type and require no runtime storage.
An MLIR type is not automatically an ABI. For example, we need IntTupe <> packed integer struct for
boundary representation.

Why not keep Fly types until convert-fly-to-rocdl?

It's possible, but it would require handling much more than function arguments. A fly value can
cross func.func, func.call, func.return, gpu..., scf... and all participants must change together.
So just using dialect conversion here.


Packing separates the compile-time structure from runtime payload.


Why pack at SCF boundaries too? Because an scf.if result or scf.for loop-carried value is also an
SSA boundary with a declared type.
*/

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include <functional>

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYREWRITEFUNCSIGNATUREPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

/*
NOTE:
Consider:
!fly.int_tuple<(?, 8, (?, 4))>
Its static leaves are 8 and 4
Only two dynamic values need to cross the boundary. The pass recursively collects them.

*/
void collectDynamicLeaves(IntTupleAttr attr, SmallVectorImpl<IntAttr> &leaves) {
  if (attr.isLeaf()) {
    // NOTE: Leaf means it is either a static int or a dynamic int.
    auto intAttr = attr.extractIntFromLeaf();
    if (!intAttr.isStatic())
      // NOTE: collect if not static
      leaves.push_back(intAttr);
  } else {
    for (int i = 0; i < attr.rank(); ++i)
      collectDynamicLeaves(attr.at(i), leaves);
  }
}

bool isStaticNarrowLayout(Attribute attr) {
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return layout.isStatic();
  if (auto composed = dyn_cast<ComposedLayoutAttr>(attr))
    return composed.isStatic();
  return true;
}

//===----------------------------------------------------------------------===//
// DSL type -> LLVM packed struct type
//
// Only dynamic sub-components generate struct fields.
// Fully static sub-components are omitted from the struct.
//===----------------------------------------------------------------------===//

/*
NOTE:
builds a packed LLVM struct, so !fly.int_tuple<(?, 8, (?, 4))> becomes {i32, i32} (the two dynamic
leaves). The static values remain encoded in the original Fly type used when reconstructing the
value.
*/
LLVM::LLVMStructType getIntTupleStructType(MLIRContext *ctx, IntTupleAttr attr) {
  SmallVector<IntAttr> leaves;
  collectDynamicLeaves(attr, leaves);
  SmallVector<Type> fields;
  fields.reserve(leaves.size());
  for (auto leaf : leaves)
    fields.push_back(IntegerType::get(ctx, leaf.getWidth()));
  return LLVM::LLVMStructType::getLiteral(ctx, fields, true);
}

/*
NOTE:
A Fly layout consists of shape and stride
The carrier includes only whichever pieces have dynamic leaves
For example:
!fly.layout<(16, ?):(1, ?)> becomes
  !llvm.struct<packed (
      struct<packed (i32)>,
      struct<packed (i32)>
  )>
The nested structs preserves the hierarchical role of those values.
If only the stride is dynamic, the carrier is !llvm.struct<packed (struct<packed (i32))>>
*/
LLVM::LLVMStructType getLayoutStructType(MLIRContext *ctx, LayoutAttr attr) {
  SmallVector<Type> fields;
  if (!attr.getShape().isStatic())
    fields.push_back(getIntTupleStructType(ctx, attr.getShape()));
  if (!attr.getStride().isStatic())
    fields.push_back(getIntTupleStructType(ctx, attr.getStride()));
  return LLVM::LLVMStructType::getLiteral(ctx, fields, true);
}

LLVM::LLVMStructType getNarrowLayoutStructType(MLIRContext *ctx, Attribute attr);
LLVM::LLVMStructType getComposedInnerStructType(MLIRContext *ctx, Attribute attr);

/*
NOTE:
A composed layout contains:
- inner transformation
- offset
- outer layout

The carrier fields are added only for dynamic components.
The pass recursively supports nested composed layouts.
A static swizzle is not serialized. It can be reconstructed from its type:
inner = StaticOp::create(builder, loc, innerTy);
A dynamic nested layout is packed recursively. This preserves complex layouts across control-flow
boundaries without carrying redundant compile-time information.
*/
LLVM::LLVMStructType getComposedLayoutStructType(MLIRContext *ctx, ComposedLayoutAttr attr) {
  SmallVector<Type> fields;
  if (!attr.isStaticOuter())
    fields.push_back(getNarrowLayoutStructType(ctx, attr.getOuter()));
  if (!attr.isStaticOffset())
    fields.push_back(getIntTupleStructType(ctx, attr.getOffset()));
  if (!attr.isStaticInner())
    // NOTE: The inner can also be a nested composed layout, so we recursively pack it.
    fields.push_back(getComposedInnerStructType(ctx, attr.getInner()));
  return LLVM::LLVMStructType::getLiteral(ctx, fields, true);
}

LLVM::LLVMStructType getComposedInnerStructType(MLIRContext *ctx, Attribute attr) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return getLayoutStructType(ctx, layoutAttr);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return getComposedLayoutStructType(ctx, composedAttr);
  // swizzle should be handled by the caller
  llvm_unreachable("unexpected inner attribute type");
}

LLVM::LLVMStructType getNarrowLayoutStructType(MLIRContext *ctx, Attribute attr) {
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return getLayoutStructType(ctx, layout);
  if (auto composed = dyn_cast<ComposedLayoutAttr>(attr))
    return getComposedLayoutStructType(ctx, composed);
  llvm_unreachable("unexpected layout attribute type");
}

/*
NOTE:
A coordinate tensor contains base coordinate and layout
*/
LLVM::LLVMStructType getCoordTensorStructType(MLIRContext *ctx, CoordTensorType ty) {
  SmallVector<Type> fields;
  if (!ty.getBase().isStatic())
    fields.push_back(getIntTupleStructType(ctx, ty.getBase()));
  if (!isStaticNarrowLayout(ty.getLayout()))
    fields.push_back(getNarrowLayoutStructType(ctx, ty.getLayout()));
  return LLVM::LLVMStructType::getLiteral(ctx, fields, true);
}

bool memrefHasDynamicLayout(fly::MemRefType ty) { return !isStaticNarrowLayout(ty.getLayout()); }

//===----------------------------------------------------------------------===//
// Pack: DSL value -> LLVM struct value (for launch_func call site)
//===----------------------------------------------------------------------===//

Value packIntTupleToStruct(OpBuilder &builder, Location loc, Value intTuple, IntTupleAttr attr,
                           LLVM::LLVMStructType structTy) {
  Value result = LLVM::UndefOp::create(builder, loc, structTy);

  int32_t structIdx = 0;
  // TODO: use get_leaves op instead of recursion
  std::function<void(Value, IntTupleAttr)> extract = [&](Value cur, IntTupleAttr curAttr) {
    if (curAttr.isLeaf()) {
      if (!curAttr.isStatic()) {
        Value scalar = GetScalarOp::create(builder, loc, cur);
        result = LLVM::InsertValueOp::create(builder, loc, structTy, result, scalar,
                                             ArrayRef<int64_t>{static_cast<int64_t>(structIdx)});
        structIdx++;
      }
      return;
    }
    for (int32_t i = 0; i < curAttr.rank(); ++i) {
      IntTupleType childTy = IntTupleType::get(curAttr.at(i));
      if (!childTy.isStatic()) {
        Value child = GetOp::create(builder, loc, childTy, cur,
                                    DenseI32ArrayAttr::get(builder.getContext(), {i}));
        extract(child, curAttr.at(i));
      }
    }
  };

  extract(intTuple, attr);
  return result;
}

/*
NOTE:
At function entry, the pass extracts the dynamic fields and builds:
%shape = fly.make_int_tuple ...
%stride = fly.make_int_tuple ...
%layout = fly.make_layout %shape, %stride

Static shape or stride components are rebuilt using an empty make_int_tuple whose result type
contains the static values: %shape = fly.make_int_tuple() : !fly.int_tuple<(16)>
*/

Value packLayoutToStruct(OpBuilder &builder, Location loc, Value layout, LayoutAttr attr,
                         LLVM::LLVMStructType structTy) {
  Value result = LLVM::UndefOp::create(builder, loc, structTy);
  int64_t idx = 0;
  if (!attr.getShape().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value shapeValue = GetShapeOp::create(builder, loc, layout);
    Value shapeStruct = packIntTupleToStruct(builder, loc, shapeValue, attr.getShape(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, shapeStruct,
                                         ArrayRef<int64_t>{idx});
    idx++;
  }
  if (!attr.getStride().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value strideValue = GetStrideOp::create(builder, loc, layout);
    Value strideStruct = packIntTupleToStruct(builder, loc, strideValue, attr.getStride(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, strideStruct,
                                         ArrayRef<int64_t>{idx});
  }
  return result;
}

Value packComposedInnerToStruct(OpBuilder &builder, Location loc, Value inner, Attribute attr,
                                LLVM::LLVMStructType innerStructTy);
Value packNarrowLayoutToStruct(OpBuilder &builder, Location loc, Value layout, Attribute attr,
                               LLVM::LLVMStructType structTy);

/*

 */
Value packComposedLayoutToStruct(OpBuilder &builder, Location loc, Value composed,
                                 ComposedLayoutAttr attr, LLVM::LLVMStructType structTy) {
  Value result = LLVM::UndefOp::create(builder, loc, structTy);
  int64_t idx = 0;
  if (!attr.isStaticOuter()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value outerVal = ComposedGetOuterOp::create(builder, loc, composed);
    Value outerStruct = packNarrowLayoutToStruct(builder, loc, outerVal, attr.getOuter(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, outerStruct,
                                         ArrayRef<int64_t>{idx});
    idx++;
  }
  if (!attr.isStaticOffset()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value offsetVal = ComposedGetOffsetOp::create(builder, loc, composed);
    Value offsetStruct = packIntTupleToStruct(builder, loc, offsetVal, attr.getOffset(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, offsetStruct,
                                         ArrayRef<int64_t>{idx});
    idx++;
  }
  if (!attr.isStaticInner()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value innerVal = ComposedGetInnerOp::create(builder, loc, composed);
    Value innerStruct = packComposedInnerToStruct(builder, loc, innerVal, attr.getInner(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, structTy, result, innerStruct,
                                         ArrayRef<int64_t>{idx});
  }
  return result;
}

Value packComposedInnerToStruct(OpBuilder &builder, Location loc, Value inner, Attribute attr,
                                LLVM::LLVMStructType innerStructTy) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return packLayoutToStruct(builder, loc, inner, layoutAttr, innerStructTy);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return packComposedLayoutToStruct(builder, loc, inner, composedAttr, innerStructTy);
  llvm_unreachable("unexpected inner attribute type");
}

Value packNarrowLayoutToStruct(OpBuilder &builder, Location loc, Value layout, Attribute attr,
                               LLVM::LLVMStructType structTy) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return packLayoutToStruct(builder, loc, layout, layoutAttr, structTy);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return packComposedLayoutToStruct(builder, loc, layout, composedAttr, structTy);
  llvm_unreachable("unexpected layout attribute type");
}

Value packCoordTensorToStruct(OpBuilder &builder, Location loc, Value operand, CoordTensorType ty,
                              Type structTy) {
  auto outerStructTy = cast<LLVM::LLVMStructType>(structTy);
  Value result = LLVM::UndefOp::create(builder, loc, outerStructTy);
  int64_t idx = 0;
  if (!ty.getBase().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(outerStructTy.getBody()[idx]);
    Value iter = GetIterOp::create(builder, loc, operand);
    Value baseStruct = packIntTupleToStruct(builder, loc, iter, ty.getBase(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, outerStructTy, result, baseStruct,
                                         ArrayRef<int64_t>{idx});
    idx++;
  }
  if (!isStaticNarrowLayout(ty.getLayout())) {
    auto fieldTy = cast<LLVM::LLVMStructType>(outerStructTy.getBody()[idx]);
    Value layout = GetLayoutOp::create(builder, loc, operand);
    Value layoutStruct = packNarrowLayoutToStruct(builder, loc, layout, ty.getLayout(), fieldTy);
    result = LLVM::InsertValueOp::create(builder, loc, outerStructTy, result, layoutStruct,
                                         ArrayRef<int64_t>{idx});
  }
  return result;
}

/*
NOTE:
A fly memref is conceptually:
MemRef = (pointer, layout)
For !fly.memref<f32, global, 32:1>, the layout is completely static, therefore only the pointer must
cross the boundary. It is packed as a single !fly.ptr carrier.
For !fly.memref<f32, global, (?, 32):1>, the layout is dynamic, so it is packed as
  !llvm.struct<packed (
      !fly.ptr,
      !llvm.struct<packed (i32)>
  )>
*/
std::pair<Value, Value> packMemRefToPtrAndLayout(OpBuilder &builder, Location loc, Value operand,
                                                 fly::MemRefType memrefTy) {
  Value ptrValue = GetIterOp::create(builder, loc, operand);
  Value layoutValue = GetLayoutOp::create(builder, loc, operand);

  if (!memrefHasDynamicLayout(memrefTy))
    // NOTE: if the layout is static, only pack ptr
    return {ptrValue, Value()};

  auto layoutStructTy = getNarrowLayoutStructType(memrefTy.getContext(), memrefTy.getLayout());
  Value layoutStruct =
      packNarrowLayoutToStruct(builder, loc, layoutValue, memrefTy.getLayout(), layoutStructTy);
  return {ptrValue, layoutStruct};
}

Value packDSLValueToStruct(OpBuilder &builder, Location loc, Value operand, Type ty,
                           LLVM::LLVMStructType structTy) {
  if (auto intTupleTy = dyn_cast<IntTupleType>(ty))
    return packIntTupleToStruct(builder, loc, operand, intTupleTy.getAttr(), structTy);
  if (auto layoutTy = dyn_cast<LayoutType>(ty))
    return packLayoutToStruct(builder, loc, operand, layoutTy.getAttr(), structTy);
  if (auto composedTy = dyn_cast<ComposedLayoutType>(ty))
    return packComposedLayoutToStruct(builder, loc, operand, composedTy.getAttr(), structTy);
  if (auto coordTy = dyn_cast<CoordTensorType>(ty))
    return packCoordTensorToStruct(builder, loc, operand, coordTy, structTy);
  llvm_unreachable("unexpected DSL type");
}

//===----------------------------------------------------------------------===//
// Unpack: LLVM struct arg -> reconstruct normal-form DSL value (for func entry)
//===----------------------------------------------------------------------===//

Value unpackIntTupleFromStruct(OpBuilder &builder, Location loc, Value structVal, IntTupleAttr attr,
                               LLVM::LLVMStructType structTy) {
  SmallVector<Value> dyncElems;
  for (size_t i = 0; i < structTy.getBody().size(); ++i) {
    Type fieldTy = structTy.getBody()[i];
    Value val = LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal,
                                             ArrayRef<int64_t>{static_cast<int64_t>(i)});
    dyncElems.push_back(val);
  }
  return MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr), dyncElems);
}

Value unpackLayoutFromStruct(OpBuilder &builder, Location loc, Value structVal, LayoutAttr attr,
                             LLVM::LLVMStructType structTy) {
  int64_t idx = 0;
  Value shape;
  if (!attr.getShape().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    shape = unpackIntTupleFromStruct(builder, loc, fieldVal, attr.getShape(), fieldTy);
    idx++;
  } else {
    shape = MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr.getShape()), ValueRange{});
  }

  Value stride;
  if (!attr.getStride().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    stride = unpackIntTupleFromStruct(builder, loc, fieldVal, attr.getStride(), fieldTy);
  } else {
    stride =
        MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr.getStride()), ValueRange{});
  }

  return MakeLayoutOp::create(builder, loc, LayoutType::get(attr), shape, stride);
}

Value unpackComposedInnerFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                    Attribute attr, LLVM::LLVMStructType innerStructTy);
Value unpackNarrowLayoutFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                   Attribute attr, LLVM::LLVMStructType structTy);

Type getNarrowLayoutType(Attribute attr) {
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return LayoutType::get(layout);
  if (auto composed = dyn_cast<ComposedLayoutAttr>(attr))
    return ComposedLayoutType::get(composed);
  llvm_unreachable("unexpected layout attribute type");
}

Value unpackComposedLayoutFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                     ComposedLayoutAttr attr, LLVM::LLVMStructType structTy) {
  int64_t idx = 0;
  Value outer;
  if (!attr.isStaticOuter()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    outer = unpackNarrowLayoutFromStruct(builder, loc, fieldVal, attr.getOuter(), fieldTy);
    idx++;
  } else {
    if (auto layout = dyn_cast<LayoutAttr>(attr.getOuter()))
      outer = LayoutType::get(layout).rebuildStaticValue(builder, loc, nullptr);
    if (auto composed = dyn_cast<ComposedLayoutAttr>(attr.getOuter()))
      outer = ComposedLayoutType::get(composed).rebuildStaticValue(builder, loc, nullptr);
  }

  Value offset;
  if (!attr.isStaticOffset()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    offset = unpackIntTupleFromStruct(builder, loc, fieldVal, attr.getOffset(), fieldTy);
    idx++;
  } else {
    offset =
        MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr.getOffset()), ValueRange{});
  }

  Value inner;
  if (!attr.isStaticInner()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(structTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    inner = unpackComposedInnerFromStruct(builder, loc, fieldVal, attr.getInner(), fieldTy);
  } else {
    auto innerAttr = attr.getInner();
    Type innerTy;
    if (auto layout = dyn_cast<LayoutAttr>(innerAttr))
      innerTy = LayoutType::get(layout);
    if (auto composed = dyn_cast<ComposedLayoutAttr>(innerAttr))
      innerTy = ComposedLayoutType::get(composed);
    if (auto swizzle = dyn_cast<SwizzleAttr>(innerAttr))
      innerTy = SwizzleType::get(swizzle);
    if (auto coordSwizzle = dyn_cast<CoordSwizzleAttr>(innerAttr))
      innerTy = CoordSwizzleType::get(coordSwizzle);
    inner = StaticOp::create(builder, loc, innerTy);
  }

  return MakeComposedLayoutOp::create(builder, loc, inner, offset, outer);
}

Value unpackComposedInnerFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                    Attribute attr, LLVM::LLVMStructType innerStructTy) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return unpackLayoutFromStruct(builder, loc, structVal, layoutAttr, innerStructTy);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return unpackComposedLayoutFromStruct(builder, loc, structVal, composedAttr, innerStructTy);
  llvm_unreachable("unexpected inner attribute type");
}

Value unpackNarrowLayoutFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                   Attribute attr, LLVM::LLVMStructType structTy) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(attr))
    return unpackLayoutFromStruct(builder, loc, structVal, layoutAttr, structTy);
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(attr))
    return unpackComposedLayoutFromStruct(builder, loc, structVal, composedAttr, structTy);
  llvm_unreachable("unexpected layout attribute type");
}

Value unpackCoordTensorFromStruct(OpBuilder &builder, Location loc, Value structVal,
                                  CoordTensorType coordTy, LLVM::LLVMStructType structTy) {
  auto outerStructTy = cast<LLVM::LLVMStructType>(structVal.getType());
  int64_t idx = 0;

  Value iter;
  if (!coordTy.getBase().isStatic()) {
    auto fieldTy = cast<LLVM::LLVMStructType>(outerStructTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    iter = unpackIntTupleFromStruct(builder, loc, fieldVal, coordTy.getBase(), fieldTy);
    idx++;
  } else {
    iter = MakeIntTupleOp::create(builder, loc, IntTupleType::get(coordTy.getBase()), ValueRange{});
  }

  Value layout;
  if (!isStaticNarrowLayout(coordTy.getLayout())) {
    auto fieldTy = cast<LLVM::LLVMStructType>(outerStructTy.getBody()[idx]);
    Value fieldVal =
        LLVM::ExtractValueOp::create(builder, loc, fieldTy, structVal, ArrayRef<int64_t>{idx});
    layout = unpackNarrowLayoutFromStruct(builder, loc, fieldVal, coordTy.getLayout(), fieldTy);
  } else {
    layout = StaticOp::create(builder, loc, getNarrowLayoutType(coordTy.getLayout()));
  }

  return MakeViewOp::create(builder, loc, iter, layout);
}

Value unpackDSLValueFromStruct(OpBuilder &builder, Location loc, Value structVal, Type oldType) {
  auto structTy = cast<LLVM::LLVMStructType>(structVal.getType());
  if (auto intTupleTy = dyn_cast<IntTupleType>(oldType))
    return unpackIntTupleFromStruct(builder, loc, structVal, intTupleTy.getAttr(), structTy);
  if (auto layoutTy = dyn_cast<LayoutType>(oldType))
    return unpackLayoutFromStruct(builder, loc, structVal, layoutTy.getAttr(), structTy);
  if (auto composedTy = dyn_cast<ComposedLayoutType>(oldType))
    return unpackComposedLayoutFromStruct(builder, loc, structVal, composedTy.getAttr(), structTy);
  if (auto coordTy = dyn_cast<CoordTensorType>(oldType))
    return unpackCoordTensorFromStruct(builder, loc, structVal, coordTy, structTy);
  llvm_unreachable("unexpected DSL type");
}

//===----------------------------------------------------------------------===//
// DslTypeConverter
//
// Drives function-signature / scf-boundary / launch_func rewriting through the
// standard DialectConversion infrastructure:
//
//   * Static DSL types (anything implementing MayStaticTypeInterface and
//     reporting isStatic()) convert to **zero** carrier types. Source
//     materialization rebuilds a `fly.static` op at the use site; target
//     materialization at boundaries drops the operand.
//   * Dynamic IntTuple / Layout / ComposedLayout / CoordTensor convert to a
//     single LLVM packed struct of their dynamic leaves. Source / target
//     materializations call into the existing unpack / pack helpers above.
//   * fly.memref expands to a `fly.ptr` carrier plus, when the layout is
//     dynamic, an LLVM struct carrier holding the dynamic layout leaves. The
//     pointer cannot live inside an llvm.struct (the LLVM verifier rejects
//     `!fly.ptr` as an llvm.struct field type), which is why we use a 1:N
//     conversion here instead of wrapping everything in a single outer struct.
//   * Everything else passes through unchanged.
//===----------------------------------------------------------------------===//

class DslTypeConverter : public TypeConverter {
public:
  DslTypeConverter(MLIRContext *ctx) {
    // Conversions are dispatched most-recently-added-first; register the
    // generic static-sink last so it has priority over the per-type expanders
    // below for any DSL type that reports isStatic().
    addConversion([](Type ty) -> std::optional<Type> { return ty; });

    addConversion([ctx](IntTupleType ty, SmallVectorImpl<Type> &out) -> LogicalResult {
      out.push_back(getIntTupleStructType(ctx, ty.getAttr()));
      return success();
    });
    addConversion([ctx](LayoutType ty, SmallVectorImpl<Type> &out) -> LogicalResult {
      out.push_back(getLayoutStructType(ctx, ty.getAttr()));
      return success();
    });
    addConversion([ctx](ComposedLayoutType ty, SmallVectorImpl<Type> &out) -> LogicalResult {
      out.push_back(getComposedLayoutStructType(ctx, ty.getAttr()));
      return success();
    });
    addConversion([ctx](CoordTensorType ty, SmallVectorImpl<Type> &out) -> LogicalResult {
      out.push_back(getCoordTensorStructType(ctx, ty));
      return success();
    });
    addConversion([ctx](fly::MemRefType ty, SmallVectorImpl<Type> &out) -> LogicalResult {
      out.push_back(ty.getPointerType());
      if (memrefHasDynamicLayout(ty))
        out.push_back(getNarrowLayoutStructType(ctx, ty.getLayout()));
      return success();
    });

    /*
    NOTE: A Fly type often stores static information: !fly.int_tuple<(4, 8)>
    Suppose a function originally has
    func.func @foo(
      %tile: !fly.int_tuple<(4, 8)>,
    )
    The argument carries no runtime information. Therefore this pass converts it to:
    func.func @foo()
    If the body needs %tile,the pass reconstructs it:
    %tile = fly.static : !fly.int_tuple<(4, 8)>
    */
    addConversion(
        [](MayStaticTypeInterface ty, SmallVectorImpl<Type> &out) -> std::optional<LogicalResult> {
          if (!ty.isStatic())
            return std::nullopt;
          // 1:0 conversion - static DSL value is sunk at the boundary.
          return success();
        });

    // Source materialization: carriers -> reconstruct an old-typed DSL value
    // for any remaining uses (function entries, scf body entries, etc.).

    /*
    NOTE:
    At function or region entry:
    %shapeMetadata = llvm.extractvalue ...
    %strideMetadata = llvm.extractvalue ...
    %shape = fly.make_int_tuple ...
    %stride = fly.make_int_tuple ...
    %layout = fly.make_layout %shape, %stride
    %memref = fly.make_view %ptr, %layout

    Downstream passes continue seeing a normal-form memref produced by fly.make_view
    */
    addSourceMaterialization(
        [](OpBuilder &b, Type oldType, ValueRange carriers, Location loc) -> Value {
          // NOTE: unpack at function or region entry
          if (carriers.empty()) {
            // 1:0 (static DSL was sunk) - rebuild a `fly.static` value so
            // downstream patterns keep seeing the normal-form sentinel.
            if (isa<MayStaticTypeInterface>(oldType))
              return StaticOp::create(b, loc, oldType);
            return Value();
          }
          if (auto memrefTy = dyn_cast<fly::MemRefType>(oldType)) {
            Value layout;
            if (carriers.size() == 2) {
              auto layoutStructTy = cast<LLVM::LLVMStructType>(carriers[1].getType());
              layout = unpackNarrowLayoutFromStruct(b, loc, carriers[1], memrefTy.getLayout(),
                                                    layoutStructTy);
            } else {
              layout = StaticOp::create(b, loc, getNarrowLayoutType(memrefTy.getLayout()));
            }
            return MakeViewOp::create(b, loc, carriers[0], layout);
          }
          if (carriers.size() == 1)
            return unpackDSLValueFromStruct(b, loc, carriers[0], oldType);
          return Value();
        });

    // Target materialization: an old-typed DSL value -> carrier(s) for the
    // new boundary representation. Called at scf yields, launch_func operands,
    // etc.

    /*
    NOTE:
    At call or launch site, use fly.get_layout or fly.get_iter and pack into a struct if dynamic.
    */
    addTargetMaterialization([](OpBuilder &b, TypeRange newTypes, ValueRange oldValues,
                                Location loc, Type) -> SmallVector<Value> {
      // NOTE: pack at a call or launch site
      if (newTypes.empty())
        return {};
      assert(oldValues.size() == 1 && "DSL boundary materialization expects a single source");
      Value oldValue = oldValues.front();
      Type oldType = oldValue.getType();
      if (auto memrefTy = dyn_cast<fly::MemRefType>(oldType)) {
        auto [ptr, layoutStruct] = packMemRefToPtrAndLayout(b, loc, oldValue, memrefTy);
        SmallVector<Value> result;
        result.push_back(ptr);
        if (layoutStruct)
          result.push_back(layoutStruct);
        return result;
      }
      auto structTy = cast<LLVM::LLVMStructType>(newTypes.front());
      return {packDSLValueToStruct(b, loc, oldValue, oldType, structTy)};
    });
  }
};

class RewriteLaunchFuncOperands : public OpConversionPattern<gpu::LaunchFuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> kernelOperands;
    for (ValueRange carriers : adaptor.getKernelOperands())
      kernelOperands.append(carriers.begin(), carriers.end());

    // NOTE: use rewriter to modify the kernel operands in place, since the number of operands may
    // change due to the type conversion
    rewriter.modifyOpInPlace(op, [&] { op.getKernelOperandsMutable().assign(kernelOperands); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class RewriteFuncSignaturePass
    : public mlir::fly::impl::FlyRewriteFuncSignaturePassBase<RewriteFuncSignaturePass> {
public:
  using mlir::fly::impl::FlyRewriteFuncSignaturePassBase<
      RewriteFuncSignaturePass>::FlyRewriteFuncSignaturePassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    DslTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    // NOTE: install the standard function signature conversion patterns for func.func, func.return,
    // and func.call
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(patterns, typeConverter);

    /*
    NOTE: we also need to convert the return because originally
    func.func @callee() -> !fly.memref<f32, global, (?, 32):1>
    But we need
    func.func @callee() -> (!fly.ptr, !llvm.struct<packed (i32)>)
    */
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // Everything not explicitly handled below is legal
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getOperandTypes()) &&
             typeConverter.isLegal(op.getResultTypes());
    });
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>([&](gpu::LaunchFuncOp op) {
      return llvm::all_of(op.getKernelOperands().getTypes(),
                          [&](Type t) { return typeConverter.isLegal(t); });
    });

    // scf.{for,if,while} + scf.yield + scf.condition signature handling.
    /*
    NOTE: we also convert for scf.{for,if,while} because region boundaries behave like function
    boundaries.
    */
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

    /*
    NOTE:
    Since kernel type can change, we need to rewrite the launch_func operands to match the new
    kernel signature.

    Before: gpu.launch_func @kernels::@kernel args(%memref : !fly.memref<...>)
    After:
    gpu.launch_func @kernels::@kernel
        args(
          %ptr : !fly.ptr<...>,
          %layout_metadata : !llvm.struct<packed (...)>
        )
    */
    patterns.add<RewriteLaunchFuncOperands>(typeConverter, ctx);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
