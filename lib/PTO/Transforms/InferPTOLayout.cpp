//===- InferPTOLayout.cpp - Infer layout for global tensor views -----------===//
//
// The pto-isa GlobalTensor ABI expects shape/stride to be represented in a 5D
// right-aligned form (pad leading dims with 1). We infer ND/DN/NZ with the same
// 5D view here and attach an optional `layout` attribute to:
//   - memref.reinterpret_cast (lowered from pto.make_tensor_view)
//   - memref.subview          (lowered from pto.partition_view)
//   - pto.tload / pto.tstore  (for fully-static GM memrefs)
//
// EmitC lowering should consume this attribute and avoid re-inferring layout
// when it is available.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static std::optional<int64_t> getConstInt(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
    return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static std::optional<int64_t> getConstInt(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return ia.getInt();
    return std::nullopt;
  }
  return getConstInt(ofr.get<Value>());
}

static unsigned elemByteSize(Type ty) {
  if (auto f = dyn_cast<FloatType>(ty))
    return f.getWidth() / 8;
  if (auto i = dyn_cast<IntegerType>(ty))
    return i.getWidth() / 8;
  return 0;
}

static bool isGlobalMemRef(MemRefType ty) {
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(ty.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    return (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  // Treat missing memory_space as GM.
  return true;
}

struct ShapeStride5D {
  SmallVector<int64_t, 5> shape;
  SmallVector<int64_t, 5> stride;
};

static std::optional<ShapeStride5D> rightAlignTo5D(ArrayRef<int64_t> shape,
                                                   ArrayRef<int64_t> stride) {
  if (shape.size() != stride.size())
    return std::nullopt;
  if (shape.size() > 5)
    return std::nullopt;

  ShapeStride5D out;
  out.shape.assign(5, 1);
  out.stride.assign(5, 1);

  const int rank = static_cast<int>(shape.size());
  const int shift = 5 - rank;
  for (int i = 0; i < rank; ++i) {
    out.shape[shift + i] = shape[i];
    out.stride[shift + i] = stride[i];
  }

  // Derive the padded leading strides with the same rule used in EmitC:
  // stride[i] = shape[i+1] * stride[i+1].
  for (int i = shift - 1; i >= 0; --i)
    out.stride[i] = out.shape[i + 1] * out.stride[i + 1];

  return out;
}

static std::optional<Layout> inferLayout5D(ArrayRef<int64_t> shape,
                                           ArrayRef<int64_t> strides,
                                           unsigned elemBytes) {
  if (shape.size() != strides.size() || elemBytes == 0)
    return std::nullopt;
  if (auto padded = rightAlignTo5D(shape, strides)) {
    auto &sh = padded->shape;
    auto &st = padded->stride;

    // NZ: 5D right-aligned, check middle dims (sh3/sh4/sh5 per spec)
    int64_t sh3 = sh[2], sh4 = sh[3], sh5 = sh[4];
    int64_t st4 = st[3], st5 = st[4];
    bool alignMatch = (sh3 == 16) && (sh3 * sh4 * elemBytes == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch)
      return Layout::NZ;

    // ND/DN are minor-2D layout hints for the last two dimensions
    // (DIM_3 = rows, DIM_4 = cols). They are not a full 5D row/col-major tag.
    //
    // For vector-like shapes where one minor dim is 1, multiple stride patterns
    // are semantically equivalent. Prefer:
    //   - DN when cols == 1 (column vector)
    //   - ND when rows == 1 (row vector)
    const int64_t rows = sh[3];
    const int64_t cols = sh[4];
    const int64_t rowStride = st[3];
    const int64_t colStride = st[4];

    bool nd = true;
    if (cols != 1 && colStride != 1)
      nd = false;
    if (rows != 1) {
      if (cols == 1) {
        nd &= (rowStride == 1);
      } else {
        nd &= (rowStride == cols);
      }
    }

    bool dn = true;
    if (rows != 1 && rowStride != 1)
      dn = false;
    if (cols != 1) {
      if (rows == 1) {
        dn &= (colStride == 1);
      } else {
        dn &= (colStride == rows);
      }
    }

    if (nd && dn) {
      if (cols == 1 && rows != 1)
        return Layout::DN;
      return Layout::ND;
    }
    if (dn)
      return Layout::DN;
    if (nd)
      return Layout::ND;

    return Layout::ND; // fallback
  }
  return std::nullopt;
}

struct InferPTOLayoutPass
    : public PassWrapper<InferPTOLayoutPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferPTOLayoutPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto setLayout = [&](Operation *op, Layout layout) {
      op->setAttr("layout", LayoutAttr::get(op->getContext(), layout));
    };

    auto verifyOrSetLayout = [&](Operation *op,
                                 std::optional<Layout> inferred) -> void {
      auto existing = op->getAttrOfType<LayoutAttr>("layout");
      if (existing) {
        if (inferred && existing.getLayout() != *inferred) {
          op->emitError() << "layout mismatch: user-specified layout="
                          << stringifyLayout(existing.getLayout())
                          << " but inferred=" << stringifyLayout(*inferred);
          signalPassFailure();
        }
        return;
      }
      setLayout(op, inferred.value_or(Layout::ND));
    };

    // ------------------------------------------------------------------
    // 1) pto.make_tensor_view (only if it still exists in the pipeline)
    // ------------------------------------------------------------------
    func.walk([&](MakeTensorViewOp op) {
      auto tvTy = dyn_cast<TensorViewType>(op.getResult().getType());
      if (!tvTy)
        return;

      const size_t rank = op.getShape().size();
      if (rank == 0 || rank > 5) {
        verifyOrSetLayout(op.getOperation(), std::nullopt);
        return;
      }

      SmallVector<int64_t> shape;
      shape.reserve(rank);
      for (size_t i = 0; i < rank; ++i) {
        int64_t dim = tvTy.getShape()[i];
        if (dim != ShapedType::kDynamic) {
          shape.push_back(dim);
          continue;
        }
        auto v = getConstInt(op.getShape()[i]);
        if (!v) {
          verifyOrSetLayout(op.getOperation(), std::nullopt);
          return;
        }
        shape.push_back(*v);
      }

      SmallVector<int64_t> strides;
      strides.reserve(rank);
      for (Value s : op.getStrides()) {
        auto v = getConstInt(s);
        if (!v) {
          verifyOrSetLayout(op.getOperation(), std::nullopt);
          return;
        }
        strides.push_back(*v);
      }

      verifyOrSetLayout(
          op.getOperation(),
          inferLayout5D(shape, strides, elemByteSize(tvTy.getElementType())));
    });

    // ------------------------------------------------------------------
    // 2) memref.reinterpret_cast (lowered from make_tensor_view)
    // ------------------------------------------------------------------
    func.walk([&](memref::ReinterpretCastOp op) {
      auto mrTy = dyn_cast<MemRefType>(op.getType());
      if (!mrTy || !isGlobalMemRef(mrTy))
        return;

      const size_t rank = op.getMixedSizes().size();
      if (rank == 0 || rank > 5) {
        verifyOrSetLayout(op.getOperation(), std::nullopt);
        return;
      }

      SmallVector<int64_t> shape;
      shape.reserve(rank);
      for (OpFoldResult s : op.getMixedSizes()) {
        auto v = getConstInt(s);
        if (!v) {
          verifyOrSetLayout(op.getOperation(), std::nullopt);
          return;
        }
        shape.push_back(*v);
      }

      SmallVector<int64_t> strides;
      strides.reserve(rank);
      for (OpFoldResult s : op.getMixedStrides()) {
        auto v = getConstInt(s);
        if (!v) {
          verifyOrSetLayout(op.getOperation(), std::nullopt);
          return;
        }
        strides.push_back(*v);
      }

      verifyOrSetLayout(
          op.getOperation(),
          inferLayout5D(shape, strides, elemByteSize(mrTy.getElementType())));
    });

    // ------------------------------------------------------------------
    // 3) memref.subview: layout is preserved from the source view
    // ------------------------------------------------------------------
    func.walk([&](memref::SubViewOp op) {
      auto resTy = dyn_cast<MemRefType>(op.getType());
      if (!resTy || !isGlobalMemRef(resTy))
        return;

      if (op->getAttrOfType<LayoutAttr>("layout"))
        return;

      if (Operation *def = op.getSource().getDefiningOp()) {
        if (auto srcLayout = def->getAttrOfType<LayoutAttr>("layout")) {
          op->setAttr("layout", srcLayout);
          return;
        }
      }

      // Fallback: if source memref type is fully static, infer from it.
      auto srcTy = dyn_cast<MemRefType>(op.getSource().getType());
      if (!srcTy || !srcTy.hasStaticShape()) {
        setLayout(op.getOperation(), Layout::ND);
        return;
      }

      SmallVector<int64_t> strideInts;
      int64_t offset = ShapedType::kDynamic;
      if (failed(getStridesAndOffset(srcTy, strideInts, offset)) ||
          offset == ShapedType::kDynamic ||
          llvm::any_of(strideInts,
                       [](int64_t s) { return s == ShapedType::kDynamic; })) {
        setLayout(op.getOperation(), Layout::ND);
        return;
      }

      auto inferred = inferLayout5D(srcTy.getShape(), strideInts,
                                    elemByteSize(srcTy.getElementType()));
      setLayout(op.getOperation(), inferred.value_or(Layout::ND));
    });

    // ------------------------------------------------------------------
    // 4) pto.tload / pto.tstore: attach layout for static GM memrefs so EmitC
    //    doesn't need to infer again in buildGlobalTensorFromMemref().
    // ------------------------------------------------------------------
    auto inferFromStaticMemRefTy = [&](MemRefType mrTy) -> std::optional<Layout> {
      if (!mrTy.hasStaticShape() || mrTy.getRank() == 0 || mrTy.getRank() > 5)
        return std::nullopt;
      SmallVector<int64_t> strideInts;
      int64_t offset = ShapedType::kDynamic;
      if (failed(getStridesAndOffset(mrTy, strideInts, offset)))
        return std::nullopt;
      if (offset == ShapedType::kDynamic ||
          llvm::any_of(strideInts,
                       [](int64_t s) { return s == ShapedType::kDynamic; }))
        return std::nullopt;
      return inferLayout5D(mrTy.getShape(), strideInts,
                           elemByteSize(mrTy.getElementType()));
    };

    func.walk([&](pto::TLoadOp op) {
      if (op->getAttrOfType<LayoutAttr>("layout"))
        return;
      auto srcTy = dyn_cast<MemRefType>(op.getSrc().getType());
      if (!srcTy || !isGlobalMemRef(srcTy))
        return;
      setLayout(op.getOperation(),
                inferFromStaticMemRefTy(srcTy).value_or(Layout::ND));
    });

    func.walk([&](pto::TStoreOp op) {
      if (op->getAttrOfType<LayoutAttr>("layout"))
        return;
      auto dstTy = dyn_cast<MemRefType>(op.getDst().getType());
      if (!dstTy || !isGlobalMemRef(dstTy))
        return;
      setLayout(op.getOperation(),
                inferFromStaticMemRefTy(dstTy).value_or(Layout::ND));
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createInferPTOLayoutPass() {
  return std::make_unique<InferPTOLayoutPass>();
}
