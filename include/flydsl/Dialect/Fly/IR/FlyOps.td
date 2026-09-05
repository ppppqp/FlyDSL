// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

/*
NOTE: Useful mental grouping:
IntTuple operations: describe shapes, strides, and coordinates
Layout operations: map logical coordinates to linear offsets
Tensor-view operations: combine an iterator/pointer with a layout
Tiled-copy/MMA operations: distribute tensor regions over threads and instructions
Memory operations: eventually become LLVM/ROCDL loads/stores and pointers


  Operation family                  Main eliminating pass
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Shape/stride constructors         fly-canonicalize / fly-layout-lowering
  ────────────────────────────────  ──────────────────────────────────────────────────
   Tuple arithmetic                  fly-layout-lowering
  ────────────────────────────────  ──────────────────────────────────────────────────
   get, slice, group, etc.           fly-layout-lowering
  ────────────────────────────────  ──────────────────────────────────────────────────
   Layout algebra                    fly-layout-lowering
  ────────────────────────────────  ──────────────────────────────────────────────────
   Tiled partitioning                fly-layout-lowering
  ────────────────────────────────  ──────────────────────────────────────────────────
   High-level copy and gemm          fly-layout-lowering
  ────────────────────────────────  ──────────────────────────────────────────────────
   Register fragment loads/stores    fly-layout-lowering, then register SSA promotion
  ────────────────────────────────  ──────────────────────────────────────────────────
  Stateful atom calls               fly-convert-atom-call-to-ssa-form
  ────────────────────────────────  ──────────────────────────────────────────────────
   Register-backed atom operands     fly-promote-regmem-to-vectorssa
  ────────────────────────────────  ──────────────────────────────────────────────────
   Fly pointers and atom calls       convert-fly-to-rocdl
  ────────────────────────────────  ──────────────────────────────────────────────────
   GPU intrinsics                    upstream convert-gpu-to-rocdl
  ────────────────────────────────  ──────────────────────────────────────────────────
   Remaining vector/arith/func       upstream LLVM conversion passes
*/



#ifndef FLY_OPS
#define FLY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

include "flydsl/Dialect/Fly/IR/FlyDialect.td"
include "flydsl/Dialect/Fly/IR/FlyTypeDefs.td"
include "flydsl/Dialect/Fly/IR/FlyAttrDefs.td"

//===----------------------------------------------------------------------===//
// Type aliases
//===----------------------------------------------------------------------===//


// NOTE: iterator-like includes int tuple and pointer. int tuple can act as iterator for a coordinate tensor
def Fly_IteratorLikeType : AnyTypeOf<[Fly_IntTuple, Fly_Pointer]>;

// NOTE: memref refers to actual storage. CoordTensor generates coordinates rather than reading allocated memory
def Fly_TensorLikeType : AnyTypeOf<[Fly_CoordTensor, Fly_MemRef]>;

// NOTE: many layout transformation operate uniformly on
// - a bare layout
// - a swizzle/composed layout
// - a memory tensor
// - a coordinate tensor
// for tensor inputs, the operation usually preserves the iterator and transforms only the layout.
// That is why code like this works:
/*
  fx.flat_divide(A, tile)
  fx.flat_divide(idC, tile)
  even though A is a memref and idC is a layout. The result of both operations is a layout.
*/

def Fly_NarrowLayoutType : AnyTypeOf<[Fly_Layout, Fly_ComposedLayout]>;
def Fly_LayoutLikeType : AnyTypeOf<[Fly_NarrowLayoutType, Fly_TensorLikeType]>;
def Fly_IntTupleLikeType : AnyTypeOf<[Fly_IntTuple, Fly_LayoutLikeType]>;

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

def Fly_SimpleLayoutMemRef : Constraint<
  CPred<"isa<fly::LayoutAttr>(cast<fly::MemRefType>($0.getType()).getLayout())">>;
def Fly_SimpleLayoutCoordTensor : Constraint<
  CPred<"isa<fly::LayoutAttr>(cast<fly::CoordTensorType>($0.getType()).getLayout())">>;

def Fly_ComposedLayoutMemRef : Constraint<Neg<Fly_SimpleLayoutMemRef.predicate>>;
def Fly_ComposedLayoutCoordTensor : Constraint<Neg<Fly_SimpleLayoutCoordTensor.predicate>>;

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//


/*
NOTE: Materializes a value known statically from its type or attributes
FlyDSL types often encode static information directly:
- shape
- stride
- layout
- tile
- atom description
In python, it's corresponds to static(...)
e.g. return static(self.type.tile_mn)
*/
def Fly_StaticOp : Fly_Op<"static", [Pure]> {  
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

/*
NOTE: The four ops below all produce fly.inte_tuple, but preserve different semantic intent at the python level

For example:
fx.make_shape(M, N)
fx.make_stride(N, 1)
fx.make_coord(m, n)

Their distinction helps APIs and diagnostics, while canonicalization can normalize them toward common representations.
e.g. IntTuple can mix static and dynamic values, (M, 64)
*/

def Fly_MakeIntTupleOp : Fly_Op<"make_int_tuple", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_MakeShapeOp : Fly_Op<"make_shape", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_MakeStrideOp : Fly_Op<"make_stride", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_MakeCoordOp : Fly_Op<"make_coord", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
}

/*
NOTE: A layout is a coordinate-to-index function.
input: (4, 8):(8, 1)
*/
def Fly_MakeLayoutOp : Fly_Op<"make_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape, Fly_IntTuple:$stride);
  let results = (outs Fly_Layout:$result);
}

def Fly_MakeLayoutLikeOp : Fly_Op<"make_layout_like", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$ref);
  let results = (outs Fly_Layout:$result);
}

/*
NOTE: This computes packed strides from a shape and mode ordering.
fx.make_ordered_layout(shape=(4, 8), order=(1, 0)) => layout=(4, 8):(1, 4)
fx.make_ordered_layout(shape=(4, 8), order=(0, 1)) => layout=(4, 8):(8, 1)
fly-layout-lowering replaces make_ordered_layout with the concrete layout it computes.
*/
def Fly_MakeOrderedLayoutOp : Fly_Op<"make_ordered_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape, Fly_IntTuple:$order);
  let results = (outs Fly_Layout:$result);
}
/*
NOTE: A composed layout represents a transformation layered on another layout:
inner transformation --applied through--> outer coordinate-to-offset mapping
The inner value can be:
- normal layout
- swizzle
- coordinate swizzle
- another composed layout

Common GPU use: LDS layout with a swizzle
logical coordinate --outer layout--> linear LDS coordinate --inner swizzle--> bank conflict reduced address
*/
def Fly_MakeComposedLayoutOp : Fly_Op<"make_composed_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_Layout, Fly_Swizzle, Fly_CoordSwizzle, Fly_ComposedLayout]>:$inner,
                       Fly_IntTuple:$offset,
                       Fly_NarrowLayoutType:$outer);
  let results = (outs Fly_ComposedLayout:$result);
}

/*
NOTE: An identity layout preserves logical coordinate structure rather than mapping it immediately to an ordinary memory offset.
idC = fx.make_view((0, 0), fx.make_identity_layout((M, N)))
The iterator is an integer coordinate (0, 0), not a memory pointer. Therefore make_view produces a coordinate tensor.
idC[m, n] = (m, n)
This lets the kernel partition coordinates exactly like it partitions data:
thr_cC = thr_copy.partition_S(cC)
Then generate bound predicates: fx.elem_less(thr_cC[a], (M, N))
TODO(qiping): what does this mean?
I think it means thread local logical index is identity mapped to memory address (register)
*/
def Fly_MakeIdentityLayoutOp : Fly_Op<"make_identity_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape);
  let results = (outs Fly_Layout:$result);
}


/*
NOTE: 
This is the central tensor abstraction:
tensor view = iterator + layout
if iterator is a pointer, then pointer + layout = fly.memref
if iterator is a integer tuple, then integer tuple + layout = fly.coord_tensor
conceptually, accessing coordinate c means:
offset = crd2idx(c, layout)
result = advance(iter, offset)

^For a memory tensor, that gives an address. For a coordinate tensor, that gives a coordinate.

make_view does not allocate or load anything. During convert-fly-to-rocdl, memory views are lowered to
pointer-oriented representations.

*/
def Fly_MakeViewOp : Fly_Op<"make_view", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IteratorLikeType:$iter, Fly_NarrowLayoutType:$layout);
  let results = (outs Fly_TensorLikeType:$result);
}

/*
NOTE: this creates a register-memory fragment whose layout matches a source view
In vector add:
thr_rA = fx.make_fragment_like(thr_gA)
thr_rB = fx.make_fragment_like(thr_gB)
thr_rC = fx.make_fragment_like(thr_gC)
thr_pC = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)
It means: create register storage shaped like this per-thread view.

Later, fly-layout-lowering normalizes the fragment allocation
fly-promote-regmem-to-vectorssa promotes register-memory load/stores to vector SSA
*/

def Fly_MakeFragmentLikeOp : Fly_Op<"make_fragment_like", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_Layout, Fly_MemRef, Fly_CoordTensor]>:$src, OptionalAttr<TypeAttr>:$dtype);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $src (`,` $dtype^)? `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeFragmentLayoutLikeOp : Fly_Op<"make_fragment_layout_like", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_Layout, Fly_MemRef]>:$src);
  let results = (outs Fly_Layout:$result);
}

//===----------------------------------------------------------------------===//
// Extractors
//===----------------------------------------------------------------------===//

def Fly_GetScalarOp : Fly_Op<"get_scalar", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$int_tuple);
  let results = (outs AnyType:$result);
}

def Fly_GetLeavesOp : Fly_Op<"get_leaves", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$input, DefaultValuedAttr<BoolAttr, "false">:$dynamicOnly);
  let results = (outs Variadic<AnyTypeOf<[I32, I64]>>:$results);
}


/* NOTE:
A.shape emits fly.get_shape
*/

def Fly_GetShapeOp : Fly_Op<"get_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_GetStrideOp : Fly_Op<"get_stride", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_GetLayoutOp : Fly_Op<"get_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TensorLikeType:$memref);
  let results = (outs Fly_NarrowLayoutType:$result);
}

def Fly_GetIterOp : Fly_Op<"get_iter", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TensorLikeType:$memref);
  let results = (outs Fly_IteratorLikeType:$result);
}


def Fly_ComposedGetInnerOp : Fly_Op<"composed_get_inner", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs AnyType:$result);
}

def Fly_ComposedGetOffsetOp : Fly_Op<"composed_get_offset", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_ComposedGetOuterOp : Fly_Op<"composed_get_outer", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs Fly_NarrowLayoutType:$result);
}

//===----------------------------------------------------------------------===//
// IntTuple operations
//===----------------------------------------------------------------------===//

class Fly_IntTupleUnaryOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$input);
  let results = (outs Fly_IntTuple:$result);
}

class Fly_IntTupleBinaryOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$lhs, Fly_IntTuple:$rhs);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_IntTupleAddOp : Fly_IntTupleBinaryOp<"int_tuple_add">;
def Fly_IntTupleSubOp : Fly_IntTupleBinaryOp<"int_tuple_sub">;
def Fly_IntTupleMulOp : Fly_IntTupleBinaryOp<"int_tuple_mul">;
def Fly_IntTupleDivOp : Fly_IntTupleBinaryOp<"int_tuple_div">;
def Fly_IntTupleModOp : Fly_IntTupleBinaryOp<"int_tuple_mod">;


/*
NOTE:
Flattens all leaves into one product:
((2, 4), 8) => 2 * 4 * 8 = 64
*/
def Fly_IntTupleProductOp     : Fly_IntTupleUnaryOp<"int_tuple_product">;

/*
NOTE:
Computes a product per top-level mode:
((2, 4), (8, 2)) -> (8, 16)
This is used when converting hierarchical layouts into physical tile dimensions
*/
def Fly_IntTupleProductEachOp : Fly_IntTupleUnaryOp<"int_tuple_product_each">;
def Fly_IntTupleProductLikeOp : Fly_IntTupleBinaryOp<"int_tuple_product_like">;

def Fly_ShapeDivOp : Fly_IntTupleBinaryOp<"shape_div">;

/*
NOTE:
for fx.ceil_div((M, N), (8, 64))
the conceptual result is:
(
  ceil(M / 8),
  ceil(N / 64)
)
*/
def Fly_CeilDivOp  : Fly_IntTupleBinaryOp<"ceil_div">;

/*
NOTE:
recrusively compares tuple leaves and combines them into one Boolean
fx.elem_less((M, N), (8, 64)) => M < 8 and N < 64
*/
def Fly_ElemLessOp : Fly_Op<"elem_less", [Pure]> {
  let arguments = (ins Fly_IntTuple:$lhs, Fly_IntTuple:$rhs);
  let results = (outs I1:$result);
}
def Fly_EqualOp : Fly_Op<"equal", [Pure]> {
  let arguments = (ins Fly_IntTuple:$lhs, Fly_IntTuple:$rhs);
  let results = (outs I1:$result);
}

//===----------------------------------------------------------------------===//
// IntTupleLike operations
//===----------------------------------------------------------------------===//

/*
NOTE:
Selects one nested mode path. Python indexing such as:
layout[0]
layout[(1, 0)]
eventually becomes a fly.get operation.
*/
def Fly_GetOp : Fly_Op<"get", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$input, DenseI32ArrayAttr:$mode);
  let results = (outs Fly_IntTupleLikeType:$result);
}

/*
NOTE:
Selects a contiguous range of modes.
e.g. layout[1:3] becomes fly.take(layout, 1, 3)
*/
def Fly_TakeOp : Fly_Op<"take", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$tuple, I32Attr:$begin, I32Attr:$end);
  let results = (outs Fly_IntTupleLikeType:$result);
}

/*
NOTE:
Selects arbitrary modes
e.g. layout[[0, 2]] becomes fly.select(layout, [0, 2])
*/
def Fly_SelectOp : Fly_Op<"select", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$tuple, DenseI32ArrayAttr:$indices);
  let results = (outs Fly_IntTupleLikeType:$result);
}

/*
NOTE:
Combines several modes into a nested group.
e.g. (a, b, c, d) -> (a, (b, c), d)
*/
def Fly_GroupOp : Fly_Op<"group", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$tuple, I32Attr:$begin, I32Attr:$end);
  let results = (outs Fly_IntTupleLikeType:$result);
}

def Fly_AppendOp : Fly_Op<"append", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_NarrowLayoutType]>:$tuple,
                       AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$elem,
                       OptionalAttr<I32Attr>:$n);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_NarrowLayoutType]>:$result);
  let assemblyFormat = "(`<` $n^ `>`)? `(` operands `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_PrependOp : Fly_Op<"prepend", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_NarrowLayoutType]>:$tuple,
                       AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$elem,
                       OptionalAttr<I32Attr>:$n);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_NarrowLayoutType]>:$result);
  let assemblyFormat = "(`<` $n^ `>`)? `(` operands `)` attr-dict `:` functional-type(operands, results)";
}

/*
NOTE:
slice fixes selected coordinates and retains modes marked None.
e.g. tensor[None, None, bid_x, bid_y] means: retain the first two modes, and fix the last two modes to the values of bid_x and bid_y.
Slicing a tensor computes:
- an iterator offset for fixed modes
- a reduced layout for retained modes

This is how vector selects one block tile after flat_divide
*/
def Fly_SliceOp : Fly_Op<"slice", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$src, Fly_IntTuple:$coord);
  let results = (outs Fly_IntTupleLikeType:$result);
}

/*
NOTE:
dice performs a related structural operation on integer tuples and bare layouts. Roughly, it removes or projects
modes according to a coordinate profile without applying tensor iterator semantics.
TODO(qiping): what does that suppose to mean?
*/
def Fly_DiceOp : Fly_Op<"dice", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$src, Fly_IntTuple:$coord);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$result);
}

//===----------------------------------------------------------------------===//
// LayoutLike operations
//===----------------------------------------------------------------------===//

def Fly_SizeOp : Fly_Op<"size", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$int_tuple);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_CoprofileOp : Fly_Op<"coprofile", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_CoshapeOp : Fly_Op<"coshape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_CosizeOp : Fly_Op<"cosize", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}


/*
NOTE:
The mathematical heart of a layout:
coordinate + layout -> offset
*/
def Fly_Crd2IdxOp : Fly_Op<"crd2idx", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$coord, AnyTypeOf<[Fly_NarrowLayoutType, Fly_Swizzle, Fly_CoordSwizzle]>:$layout);
  let results = (outs Fly_IntTuple:$index);
}
/*
NOTE: inverse mapping for a plain layout
TODO(qiping): what about swizzles and composed layouts? Do we need to implement crd2idx for them too?
Also, WHY is the result "index" instead of "coord"?
*/
def Fly_Idx2CrdOp : Fly_Op<"idx2crd", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$coord, Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$index);
}
def Fly_GetFlatCoordOp : Fly_Op<"get_flat_coord", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$index, Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_Get1DCoordOp : Fly_Op<"get_1d_coord", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$index, Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$result);
}


/*
NOTE:
Combines compatible adjacent layout modes. For a contiguous layout (4, 8):(1, 4)
coalescing may produces: shape=(32, ) stride=(1, )
it simplifies layout structure without changing the underlying coordinate-to-offset mapping.
*/
def Fly_CoalesceOp : Fly_Op<"coalesce", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout, Optional<Fly_IntTuple>:$pattern);
  let results = (outs Fly_LayoutLikeType:$result);
  let assemblyFormat = "`(` $layout (`,` $pattern^)? `)` attr-dict `:` functional-type(operands, results)";
}

/*
NOTE:
It maps the coordinate domain of one layout through another
This is used when constructing  thread-value mappings:
layout_tv = composition(
  right_inverse(layout_mn),
  tmp
)
*/
def Fly_CompositionOp : Fly_Op<"composition", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$outer, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$inner);
  let results = (outs Fly_LayoutLikeType:$result);
}

/*
NOTE:
Constructs the part of an index space not covered by a layout, up to a requested codomain size.
It's useful for building tiling layouts.
*/
def Fly_ComplementOp : Fly_Op<"complement", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$layout, Optional<Fly_IntTuple>:$codomain_size);
  let results = (outs Fly_NarrowLayoutType:$result);
  let assemblyFormat = "`(` $layout (`,` $codomain_size^)? `)` attr-dict `:` functional-type(operands, results)";
}

/*
NOTE:
Constructs inverse mappings subject to the layout's domain/codomain properties.
right_inverse is central to converting a physical tile layout into a (thread, value) mapping
*/
def Fly_RightInverseOp : Fly_Op<"right_inverse", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$layout);
  let results = (outs Fly_NarrowLayoutType:$result);
}
def Fly_LeftInverseOp : Fly_Op<"left_inverse", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$layout);
  let results = (outs Fly_NarrowLayoutType:$result);
}

class Fly_LayoutDivideOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$divisor);
  let results = (outs Fly_LayoutLikeType:$result);
}
class Fly_LayoutProductOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$layout, Fly_Layout:$tile);
  let results = (outs Fly_NarrowLayoutType:$result);
}

def Fly_LogicalDivideOp : Fly_LayoutDivideOp<"logical_divide">;
def Fly_ZippedDivideOp : Fly_LayoutDivideOp<"zipped_divide">;
def Fly_TiledDivideOp : Fly_LayoutDivideOp<"tiled_divide">;
def Fly_FlatDivideOp : Fly_LayoutDivideOp<"flat_divide">;

def Fly_LogicalProductOp : Fly_LayoutProductOp<"logical_product">;
def Fly_ZippedProductOp : Fly_LayoutProductOp<"zipped_product">;
def Fly_TiledProductOp : Fly_LayoutProductOp<"tiled_product">;
def Fly_FlatProductOp : Fly_LayoutProductOp<"flat_product">;
def Fly_BlockedProductOp : Fly_LayoutProductOp<"blocked_product">;
def Fly_RakedProductOp : Fly_LayoutProductOp<"raked_product">;


/*
NOTE: This changes a layout to account for reinterpreting storage with a different element width
For example, reinterpreting four 8-bit elements as one 32-bit element changes:
- logical element count
- appropriate strides
- vectorization granularity
*/
def Fly_RecastLayoutOp : Fly_Op<"recast_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins I32Attr:$new_type_bits, I32Attr:$old_type_bits, Fly_NarrowLayoutType:$src);
  let results = (outs Fly_NarrowLayoutType:$result);
}

def Fly_TileToShapeOp : Fly_Op<"tile_to_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$block, Fly_IntTuple:$trg_shape, Fly_IntTuple:$ord_shape);
  let results = (outs Fly_NarrowLayoutType:$result);
}

//===----------------------------------------------------------------------===//
// Atom and Tiled Mma/Copy ops
//===----------------------------------------------------------------------===//


/*
NOTE:
Creates an MMA atom whose concrete type identifies an MFMA or WMMA operation and its operand layouts.
The operation has no operands, because the instruction description is carried by the result type.
*/
def Fly_MakeMmaAtomOp : Fly_Op<"make_mma_atom", [Pure]> {
  let arguments = (ins);
  let results = (outs Fly_MmaAtom:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}


/*
NOTE:
An atom describes one architecture-level or abstract primitive operation.
For vector add:
fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
produces an atom describing a 128-bit transfer interpreted as float32 values.
The atom type carries information such as:
- operation kind
- value width
- source thread-value layout
- destination thread-value layout
- reference layout
*/
def Fly_MakeCopyAtomOp : Fly_Op<"make_copy_atom", [Pure]> {
  let arguments = (ins I32Attr:$valBits);
  let results = (outs Fly_CopyAtom:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}


/*
NOTE:
Updates a configurable atom field while preserving the atom's overall type/category.
This supports instruction parameters that are supplied as values rather than fixed entirely in the base atom type.
*/
def Fly_AtomSetValueOp : Fly_Op<"atom.set_value", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyType:$atom, StrAttr:$field, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $atom `,` $field `,` $value `)` attr-dict `:` functional-type(operands, results)";
}


/*
This is the initial memory-oriented representation:
copy from source fragment/view
to destination fragment/view
optionally under this predicate
*/
def Fly_CopyAtomCall : Fly_Op<"copy_atom_call"> {
  let arguments = (ins Fly_CopyAtom:$copyAtom, Fly_MemRef:$src, Fly_MemRef:$dst, Optional<Fly_MemRef>:$pred);
}
def Fly_MmaAtomCall : Fly_Op<"mma_atom_call"> {
  let arguments = (ins Fly_MmaAtom:$mmaAtom, Fly_MemRef:$d, Fly_MemRef:$a, Fly_MemRef:$b, Fly_MemRef:$c);
}

/*
This is the value-oriented representation after register promotion starts
Instead of requiring all data to live in !fly.memref, it accepts and returns SSA values.
pass sequence:
copy_atom_call -> copy_atom_call_ssa -> convert-fly-to-rocdl
*/
def Fly_CopyAtomCallSSA : Fly_Op<"copy_atom_call_ssa", [AttrSizedOperandSegments]> {
  let arguments = (ins Fly_CopyAtom:$copyAtom, AnyType:$src,
                       Optional<AnyType>:$dst, Optional<AnyType>:$pred);
  let results = (outs Variadic<AnyType>:$results);
}
def Fly_MmaAtomCallSSA : Fly_Op<"mma_atom_call_ssa"> {
  let arguments = (ins Fly_MmaAtom:$mmaAtom, Optional<AnyType>:$d,
                       AnyType:$a, AnyType:$b, AnyType:$c);
  let results = (outs Variadic<AnyType>:$results);
}

/*
NOTE:
A tiled copy combines:
one copy instruction + mapping from (thread, value) to tile coordinate + overall tile dimensions
*/
def Fly_MakeTiledCopyOp : Fly_Op<"make_tiled_copy", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_CopyAtom:$copyAtom, Fly_Layout:$layoutThrVal, Fly_Tile:$tileMN);
  let results = (outs Fly_TiledCopy:$result);
}
def Fly_MakeTiledMmaOp : Fly_Op<"make_tiled_mma", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyType:$mmaAtom, Fly_Layout:$atomLayout, Optional<Fly_Tile>:$permutation);
  let results = (outs Fly_TiledMma:$result);
  let assemblyFormat = "`(` $mmaAtom `,` $atomLayout (`,` $permutation^)? `)` attr-dict `:` functional-type(operands, results)";
}


/*
NOTE:
Given the complete tiled-copy description, a source tensor and a thread ID, it returns the source view owned by that thread.
*/
def Fly_TiledCopyPartitionSrcOp : Fly_Op<"tiled_copy.partition_src", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_TensorLikeType:$src, Fly_IntTuple:$coord);
  let results = (outs Fly_TensorLikeType:$result);
}
def Fly_TiledCopyPartitionDstOp : Fly_Op<"tiled_copy.partition_dst", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_TensorLikeType:$dst, Fly_IntTuple:$coord);
  let results = (outs Fly_TensorLikeType:$result);
}

/*
NOTE:
Reinterprets an existing per-thread fragment according to the tiled copy's expected value layout.
This is useful when a register fragment was produced using one partitioning context but must be consumed by another compatible copy operation.
*/
def Fly_TiledCopyRetileOp : Fly_Op<"tiled_copy.retile", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_MemRef:$input);
  let results = (outs Fly_MemRef:$result);
}

/*
NOTE:
Partitions an A, B or C tensor for one thread:
operand_id = A | B | C
The operand ID matters because each MMA operand has a different role and fragment layout.
*/
def Fly_TiledMmaPartitionOp : Fly_Op<"tiled_mma.partition", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MmaOperandAttr:$operand_id, Fly_TiledMma:$tiled_mma,
                       Fly_TensorLikeType:$input, Fly_IntTuple:$coord);
  let results = (outs Fly_TensorLikeType:$result);
  let assemblyFormat = "`(` $operand_id `,` $tiled_mma `,` $input `,` $coord `)` attr-dict `:` functional-type(operands, results)";
}

/*
NOTE:
Creates the shape of a per-thread MMA partition without requiring an actual tensor.
TODO(qiping): what does that suppose to mean?
*/
def Fly_TiledMmaPartitionShapeOp : Fly_Op<"tiled_mma.partition_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MmaOperandAttr:$operand_id, Fly_TiledMma:$tiled_mma, Fly_IntTuple:$shape);
  let results = (outs Fly_IntTuple:$result);
}
/*
NOTE:
Creates a register fragment appropriate for an MMA operand. Optionally with a pipeline stage dimension.
These are lowered by fly-layout-lowering, while the eventual atom call is handled by the SSA and ROCDL conversion passes.
*/
def Fly_MmaMakeFragmentOp : Fly_Op<"mma.make_fragment", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MmaOperandAttr:$operand_id, Fly_TiledMma:$tiled_mma, Fly_MemRef:$input, OptionalAttr<I32Attr>:$stages);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $operand_id `,` $tiled_mma `,` $input (`,` `stages` `=` $stages^)? `)` attr-dict `:` functional-type(operands, results)";
}

/*
NOTE:
This is more abstract than copy_atom_call. It can represent copying an entire per-thread fragment that may contain
multiple atom-sized chunks.
ExpandCopyOpLowering expands it into repeated atom calls.
*/
def Fly_CopyOp : Fly_Op<"copy"> {
  let arguments = (ins AnyType:$copyAtom, Fly_MemRef:$src, Fly_MemRef:$dst, Optional<Fly_MemRef>:$pred);
  let assemblyFormat = "`(` $copyAtom `,` $src `,` $dst (`,` $pred^)? `)` attr-dict `:` functional-type(operands, results)";
}

/*
NOTE:
This represents a tiled fragment-level GEMM. It is not itself one hardware MMA instruction.
ExpandGemmOpLowering examines the fragment layouts, determines the necessary MMA traversal, slices A, B, C and D into atom-sized fragments
and emit a sequence of mma_atom_call operations
*/
def Fly_GemmOp : Fly_Op<"gemm"> {
  let arguments = (ins AnyType:$mmaAtom, Fly_MemRef:$d, Fly_MemRef:$a, Fly_MemRef:$b, Fly_MemRef:$c,
                       OptionalAttr<Fly_GemmTraversalOrderAttr>:$traversalOrder,
                       Optional<Fly_Layout>:$traversalLayout);
  let assemblyFormat = "`(` $mmaAtom `,` $d `,` $a `,` $b `,` $c `)` (`traversalLayout` `=` $traversalLayout^)? "
                       "attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// MemRef and Ptr operations
//===----------------------------------------------------------------------===//

/*
NOTE:
Creates a fly pointer from backend/runtime-specific inputs.
The result type carries:
- element type
- address space
- alignment
- possibly additional pointer semantics

Note that it is not marked as Pure, because pointer construction may represent allocation
*/
def Fly_MakePtrOp : Fly_Op<"make_ptr", []> {
  let arguments = (ins Variadic<AnyType>:$args, OptionalAttr<DictionaryAttr>:$dictAttrs);
  let results = (outs Fly_Pointer:$result);
}

/*
NOTE:
Returns the base pointer of dynamically allocated shared memory.
It lowers to the GPU module's dynamic LDS global.
*/
def Fly_GetDynSharedOp : Fly_Op<"get_dyn_shared", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins);
  let results = (outs Fly_Pointer:$result);
  let assemblyFormat = "`(` `)` attr-dict `:` qualified(type($result))";
}
def Fly_IntToPtrOp : Fly_Op<"inttoptr", [Pure]> {
  let arguments = (ins AnyTypeOf<[I64, I32]>:$src);
  let results = (outs Fly_Pointer:$result);
}
def Fly_PtrToIntOp : Fly_Op<"ptrtoint", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Pointer:$ptr);
  let results = (outs AnyTypeOf<[I64, I32]>:$result);
}
def Fly_ToLLVMPtrOp : Fly_Op<"to_llvm_ptr", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Pointer:$ptr, I32Attr:$llvm_address_space);
  let results = (outs LLVM_AnyPointer:$result);
}

/*
NOTE:
Advanced either a memory pointer a memory pointer or a coordinate iterator
For a pointer, it eventually becomes LLVM GEP-style address arithmetic
For a coordinate iterator, it adds logical coordinates
*/
def Fly_AddOffsetOp : Fly_Op<"add_offset", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IteratorLikeType:$ptr, Fly_IntTuple:$offset);
  let results = (outs Fly_IteratorLikeType:$result);
}

/*
NOTE:
Transforms the pointer offset according to a swizzle description, typically for LDS bank-conflict avoidance.
This is lowered to integer pointer arithmetic. The later fly-int-swizzle-simplify pass cleans up the generated bit manipulation.
*/
def Fly_ApplySwizzleOp : Fly_Op<"apply_swizzle", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Pointer:$ptr, Fly_Swizzle:$swizzle);
  let results = (outs Fly_Pointer:$result);
}
def Fly_DecompositionOp : Fly_Op<"decomposition", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TensorLikeType:$tensor);
  let results = (outs Fly_TensorLikeType:$result);
}

def Fly_PtrLoadOp : Fly_Op<"ptr.load", []> {
  let arguments = (ins Fly_IteratorLikeType:$ptr);
  let results = (outs AnyType:$result);
}
def Fly_PtrStoreOp : Fly_Op<"ptr.store"> {
  let arguments = (ins AnyType:$value, Fly_Pointer:$ptr);
}

def Fly_RecastIterOp : Fly_Op<"recast_iter", [Pure]> {
  let arguments = (ins Fly_Pointer:$src);
  let results = (outs Fly_Pointer:$result);
}

def Fly_MemRefAllocaOp : Fly_Op<"memref.alloca", []> {
  let arguments = (ins Fly_Layout:$layout);
  let results = (outs Fly_MemRef:$result);
}
def Fly_MemRefLoadVecOp : Fly_Op<"memref.load_vec", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TensorLikeType:$memref);
  let results = (outs AnyType:$result);
}
def Fly_MemRefStoreVecOp : Fly_Op<"memref.store_vec", []> {
  let arguments = (ins AnyType:$vector, Fly_MemRef:$memref); 
  let results = (outs);
}

def Fly_MemRefLoadOp : Fly_Op<"memref.load", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TensorLikeType:$memref, Fly_IntTuple:$indices);
  let results = (outs AnyType:$result);
}
def Fly_MemRefStoreOp : Fly_Op<"memref.store", []> {
  let arguments = (ins AnyType:$value, Fly_MemRef:$memref, Fly_IntTuple:$indices);
}

//===----------------------------------------------------------------------===//
// Utility ops
//===----------------------------------------------------------------------===//

def Fly_PrintOp : Fly_Op<"print"> {
  let arguments = (ins StrAttr:$format, Variadic<AnyType>:$values);
}

def Fly_AssumeOp : Fly_Op<"assume", [Pure]> {
  let arguments = (ins AnyType:$dst, AnyType:$src);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Deprecated, removed in the future
//===----------------------------------------------------------------------===//

def Fly_ExtractAlignedPointerAsIndexOp : Fly_Op<"extract_aligned_pointer_as_index", [Pure]> {
  let summary = "Extract the raw pointer from a fly.memref";
  let arguments = (ins Fly_MemRef:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // FLY_OPS
