//===- SpmdPartition.cpp ------------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace mesh {
#define GEN_PASS_DEF_SPMDPARTITION
#include "mlir/Dialect/Mesh/Transforms/Passes.h.inc"
} // namespace mesh
} // namespace mlir

#define DEBUG_TYPE "spmd-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

namespace {

LogicalResult visitOp(Operation *op, OpBuilder &builder) {
  if (llvm::isa<ShardOp>(op))
    return success(); 
  
  SmallVector<MeshShardingAttr> operandShardings;
  SmallVector<MeshShardingAttr> resultShardings;
}

//===----------------------------------------------------------------------===//
// SpmdPartition
//===----------------------------------------------------------------------===//
struct SpmdPartition : public mesh::impl::SpmdPartitionBase<SpmdPartition> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Region &region = funcOp.getBody();
    OpBuilder builder(ctx);
    if (!region.hasOneBlock()) {
      funcOp.emitOpError() << "only one block is supported!";
      return signalPassFailure();
    }
    Block &block = region.front();

    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block))) {
      if (failed(visitOp(&op, builder)))
        return signalPassFailure();
    }
  }
};

} // namespace
