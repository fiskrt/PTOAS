// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOLOWERFRONTENDPIPEOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct FrontendPipeHandles {
  Value c2vPipe;
  Value v2cPipe;
  Operation *anchorOp = nullptr;
};

template <typename InitOpT>
static LogicalResult requireFrontendGmSlotBuffer(InitOpT initOp) {
  if (initOp.getGmSlotBuffer())
    return success();
  return initOp.emitOpError("requires 'gm_slot_buffer' when lowering to a2/a3");
}

template <typename InitOpT>
static FailureOr<Value> createFrontendPipe(InitOpT initOp, IRRewriter &rewriter,
                                           PTOArch arch, Type pipeTy,
                                           int8_t dirMask, int32_t slotNum,
                                           Value localAddr,
                                           Value peerLocalAddr = Value{}) {
  Location loc = initOp.getLoc();
  auto dirAttr = rewriter.getI8IntegerAttr(dirMask);
  auto slotSizeAttr = rewriter.getI32IntegerAttr(initOp.getSlotSize());
  auto slotNumAttr = rewriter.getI32IntegerAttr(slotNum);
  auto noSplitAttr = initOp.getNosplitAttr();

  if (arch == PTOArch::A5) {
    auto pipe = rewriter.create<InitializeL2LPipeOp>(
        loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, IntegerAttr{},
        noSplitAttr, localAddr, peerLocalAddr);
    return pipe.getPipe();
  }

  if (failed(requireFrontendGmSlotBuffer(initOp)))
    return failure();

  auto localSlotNumAttr = rewriter.getI32IntegerAttr(slotNum);
  auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
      loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr, localSlotNumAttr,
      IntegerAttr{}, noSplitAttr, initOp.getGmSlotBuffer(), localAddr,
      peerLocalAddr);
  return pipe.getPipe();
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles>
lowerSingleDirectionFrontendInit(InitOpT initOp, IRRewriter &rewriter,
                                 PTOArch arch, Type pipeTy, int8_t dirMask,
                                 Value localAddr) {
  auto pipeOr =
      createFrontendPipe(initOp, rewriter, arch, pipeTy, dirMask, /*slotNum=*/8,
                         localAddr);
  if (failed(pipeOr))
    return failure();

  FrontendPipeHandles handles;
  if (dirMask == 1)
    handles.c2vPipe = *pipeOr;
  else
    handles.v2cPipe = *pipeOr;
  handles.anchorOp = pipeOr->getDefiningOp();
  return handles;
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles>
lowerBidirectionalFrontendInit(InitOpT initOp, IRRewriter &rewriter,
                               PTOArch arch, Type pipeTy) {
  auto pipeOr = createFrontendPipe(initOp, rewriter, arch, pipeTy,
                                   /*dirMask=*/3, /*slotNum=*/4,
                                   initOp.getC2vConsumerBuf(),
                                   initOp.getV2cConsumerBuf());
  if (failed(pipeOr))
    return failure();

  FrontendPipeHandles handles;
  handles.c2vPipe = *pipeOr;
  handles.v2cPipe = *pipeOr;
  handles.anchorOp = pipeOr->getDefiningOp();
  return handles;
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles> lowerFrontendInitOp(InitOpT initOp,
                                                          IRRewriter &rewriter) {
  MLIRContext *ctx = initOp.getContext();
  auto pipeTy = PipeType::get(ctx);
  PTOArch arch = getTargetArch(initOp.getOperation());

  switch (initOp.getDirMask()) {
  case 1:
    return lowerSingleDirectionFrontendInit(initOp, rewriter, arch, pipeTy,
                                            /*dirMask=*/1,
                                            initOp.getC2vConsumerBuf());
  case 2:
    return lowerSingleDirectionFrontendInit(initOp, rewriter, arch, pipeTy,
                                            /*dirMask=*/2,
                                            initOp.getV2cConsumerBuf());
  case 3:
    return lowerBidirectionalFrontendInit(initOp, rewriter, arch, pipeTy);
  default:
    return FrontendPipeHandles{};
  }
}

struct FrontendInitOps {
  AicInitializePipeOp aicInit;
  AivInitializePipeOp aivInit;
  unsigned aicInitCount = 0;
  unsigned aivInitCount = 0;
};

static FrontendInitOps collectFrontendInitOps(func::FuncOp funcOp) {
  FrontendInitOps initOps;
  funcOp.walk([&](Operation *op) {
    if (auto init = dyn_cast<AicInitializePipeOp>(op)) {
      ++initOps.aicInitCount;
      if (!initOps.aicInit)
        initOps.aicInit = init;
      return WalkResult::advance();
    }
    if (auto init = dyn_cast<AivInitializePipeOp>(op)) {
      ++initOps.aivInitCount;
      if (!initOps.aivInit)
        initOps.aivInit = init;
    }
    return WalkResult::advance();
  });
  return initOps;
}

static LogicalResult validateFrontendInitOps(func::FuncOp funcOp,
                                             const FrontendInitOps &initOps) {
  if (initOps.aicInitCount > 1)
    return funcOp.emitOpError("requires at most one pto.aic_initialize_pipe");
  if (initOps.aivInitCount > 1)
    return funcOp.emitOpError("requires at most one pto.aiv_initialize_pipe");
  if (initOps.aicInit && initOps.aivInit) {
    return funcOp.emitOpError("cannot mix pto.aic_initialize_pipe and "
                              "pto.aiv_initialize_pipe in one function");
  }
  return success();
}

template <typename InitOpT>
static void propagateFrontendNoSplitAttr(InitOpT initOp,
                                         const FrontendPipeHandles &handles) {
  auto noSplitAttr = initOp.getNosplitAttr();
  if (!noSplitAttr)
    return;

  if (handles.anchorOp)
    handles.anchorOp->setAttr("nosplit", noSplitAttr);

  Operation *c2vOp =
      handles.c2vPipe ? handles.c2vPipe.getDefiningOp() : nullptr;
  Operation *v2cOp =
      handles.v2cPipe ? handles.v2cPipe.getDefiningOp() : nullptr;

  if (c2vOp && c2vOp != handles.anchorOp)
    c2vOp->setAttr("nosplit", noSplitAttr);
  if (v2cOp && v2cOp != handles.anchorOp && v2cOp != c2vOp)
    v2cOp->setAttr("nosplit", noSplitAttr);
}

template <typename InitOpT>
static FailureOr<FrontendPipeHandles> lowerAndEraseFrontendInit(InitOpT initOp,
                                                                IRRewriter &rewriter) {
  rewriter.setInsertionPoint(initOp);
  auto loweredOr = lowerFrontendInitOp(initOp, rewriter);
  if (failed(loweredOr))
    return failure();
  propagateFrontendNoSplitAttr(initOp, *loweredOr);
  rewriter.eraseOp(initOp);
  return *loweredOr;
}

static FailureOr<FrontendPipeHandles> lowerInitIfPresent(func::FuncOp funcOp,
                                                         IRRewriter &rewriter) {
  FrontendInitOps initOps = collectFrontendInitOps(funcOp);
  if (failed(validateFrontendInitOps(funcOp, initOps)))
    return failure();
  if (initOps.aicInit)
    return lowerAndEraseFrontendInit(initOps.aicInit, rewriter);
  if (initOps.aivInit)
    return lowerAndEraseFrontendInit(initOps.aivInit, rewriter);
  return FrontendPipeHandles{};
}

static bool hasFrontendPipeOps(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<AicInitializePipeOp, AivInitializePipeOp, TPushToAivOp, TPushToAicOp,
            TPopFromAicOp, TPopFromAivOp, TFreeFromAicOp, TFreeFromAivOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static LogicalResult lowerFrontendDataOps(func::FuncOp funcOp,
                                          const FrontendPipeHandles &handles,
                                          IRRewriter &rewriter) {
  DominanceInfo dom(funcOp);
  SmallVector<Operation *> frontendOps;
  funcOp.walk([&](Operation *op) {
    if (isa<TPushToAivOp, TPushToAicOp, TPopFromAicOp, TPopFromAivOp,
            TFreeFromAicOp, TFreeFromAivOp>(op))
      frontendOps.push_back(op);
  });

  for (Operation *op : frontendOps) {
    if (!handles.anchorOp) {
      op->emitOpError("requires a frontend initialize_pipe op in the same function");
      return failure();
    }
    if (!dom.dominates(handles.anchorOp, op)) {
      op->emitOpError(
          "requires a dominating frontend initialize_pipe op");
      return failure();
    }

    rewriter.setInsertionPoint(op);

    if (auto push = dyn_cast<TPushToAivOp>(op)) {
      if (!handles.c2vPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable C2V");
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(), handles.c2vPipe,
                                           push.getSplitAttr());
      continue;
    }

    if (auto push = dyn_cast<TPushToAicOp>(op)) {
      if (!handles.v2cPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable V2C");
        return failure();
      }
      rewriter.replaceOpWithNewOp<TPushOp>(push, push.getTile(), handles.v2cPipe,
                                           push.getSplitAttr());
      continue;
    }

    if (auto pop = dyn_cast<TPopFromAicOp>(op)) {
      if (!handles.c2vPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable C2V");
        return failure();
      }
      auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                                 pop.getTile().getType());
      if (pop.getValidRow() && pop.getValidCol()) {
        rewriter.create<SetValidShapeOp>(pop.getLoc(), decl.getTile(),
                                         pop.getValidRow(), pop.getValidCol());
      }
      rewriter.create<TPopOp>(pop.getLoc(), decl.getTile(), handles.c2vPipe,
                              pop.getSplitAttr());
      rewriter.replaceOp(pop, decl.getTile());
      continue;
    }

    if (auto pop = dyn_cast<TPopFromAivOp>(op)) {
      if (!handles.v2cPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable V2C");
        return failure();
      }
      auto decl = rewriter.create<DeclareTileOp>(pop.getLoc(),
                                                 pop.getTile().getType());
      if (pop.getValidRow() && pop.getValidCol()) {
        rewriter.create<SetValidShapeOp>(pop.getLoc(), decl.getTile(),
                                         pop.getValidRow(), pop.getValidCol());
      }
      rewriter.create<TPopOp>(pop.getLoc(), decl.getTile(), handles.v2cPipe,
                              pop.getSplitAttr());
      rewriter.replaceOp(pop, decl.getTile());
      continue;
    }

    if (auto free = dyn_cast<TFreeFromAicOp>(op)) {
      if (!handles.c2vPipe) {
        op->emitOpError(
            "requires the dominating initialize_pipe op to enable C2V");
        return failure();
      }
      rewriter.replaceOpWithNewOp<TFreeOp>(free, handles.c2vPipe,
                                           free.getSplitAttr());
      continue;
    }

    auto free = cast<TFreeFromAivOp>(op);
    if (!handles.v2cPipe) {
      op->emitOpError(
          "requires the dominating initialize_pipe op to enable V2C");
      return failure();
    }
    rewriter.replaceOpWithNewOp<TFreeOp>(free, handles.v2cPipe,
                                         free.getSplitAttr());
  }

  return success();
}

struct PTOLowerFrontendPipeOpsPass
    : public mlir::pto::impl::PTOLowerFrontendPipeOpsBase<
          PTOLowerFrontendPipeOpsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hasFrontendPipeOps(funcOp))
      return;

    IRRewriter rewriter(funcOp.getContext());
    auto loweredOr = lowerInitIfPresent(funcOp, rewriter);
    if (failed(loweredOr)) {
      signalPassFailure();
      return;
    }

    if (failed(lowerFrontendDataOps(funcOp, *loweredOr, rewriter)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerFrontendPipeOpsPass() {
  return std::make_unique<PTOLowerFrontendPipeOpsPass>();
}
