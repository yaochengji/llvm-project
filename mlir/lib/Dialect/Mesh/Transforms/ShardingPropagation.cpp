//===- ShardingPropagation.cpp ------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace mesh {
#define GEN_PASS_DEF_SHARDINGPROPAGATION
#include "mlir/Dialect/Mesh/Transforms/Passes.h.inc"
} // namespace mesh
} // namespace mlir

#define DEBUG_TYPE "sharding-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

namespace {

struct ShardingPropagationPass
    : public mesh::impl::ShardingPropagationBase<ShardingPropagationPass> {
  void runOnOperation() override { return; }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::mesh::createShardingPropagationPass() {
  return std::make_unique<ShardingPropagationPass>();
}