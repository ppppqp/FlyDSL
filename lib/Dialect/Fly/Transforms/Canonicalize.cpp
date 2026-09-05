// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

/*
Why do we need this pass?

FlyDSL has several operations that may produce the same underlying type:
fly.static
fly.make_int_tuple
fly.make_shape
fly.make_stride
fly.make_coord

All of these concepts are represented by fly.int_tuple, but later lowering should not need separate
implementations for every possible producer.
*/

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYCANONICALIZEPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

template <typename IntTupleLikeOp>
class RewriteToMakeIntTuple final : public OpRewritePattern<IntTupleLikeOp> {
  using OpRewritePattern<IntTupleLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IntTupleLikeOp op, PatternRewriter &rewriter) const override {
    /*
    NOTE: Rewrite int tuple-like operations to fly.make_int_tuple.
    */
    auto newOp = MakeIntTupleOp::create(rewriter, op.getLoc(), op.getResult().getType(),
                                        op->getOperands(), op->getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

class RebuildStaticValue : public RewritePattern {
public:
  RebuildStaticValue(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    /*
    NOTE: match every operation
    */
    if (op->getNumResults() != 1)
      // NOTE: match every operation, but only rewrite those with a single result that is a fly type
      // that can be rebuilt to a static value
      return failure();
    Type resultType = op->getResult(0).getType();
    // NOTE: check if the type implements the MayStaticTypeInterface and is fully static
    /*
    The types implementing the interface are:
    IntTupleType
    LayoutType
    ComposedLayoutType
    SwizzleType
    CoordSwizzleType
    TileType
    CoordTensorType
    TiledCopyType
    TiledMMaType
    MmaAtomType
    */
    auto mayStatic = dyn_cast<MayStaticTypeInterface>(resultType);
    if (!mayStatic || !mayStatic.isStatic())
      return failure();
    // NOTE: rebuild the static value
    Value rebuild = mayStatic.rebuildStaticValue(rewriter, op->getLoc(), op->getResult(0));
    if (!rebuild)
      return failure();

    rewriter.replaceOp(op, rebuild);
    return success();
  }
};

class FlyCanonicalizePass : public mlir::fly::impl::FlyCanonicalizePassBase<FlyCanonicalizePass> {
public:
  using mlir::fly::impl::FlyCanonicalizePassBase<FlyCanonicalizePass>::FlyCanonicalizePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // NOTE: rewrite make_shape make_stride and make_coord to make_int_tuple
    patterns.add<RewriteToMakeIntTuple<MakeShapeOp>, RewriteToMakeIntTuple<MakeStrideOp>,
                 RewriteToMakeIntTuple<MakeCoordOp>>(context);
    patterns.add<RebuildStaticValue>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
