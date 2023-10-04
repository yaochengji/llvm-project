//===- ShardingInterface.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_
#define MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

namespace mesh {

using ShardingArray = SmallVector<SmallVector<int64_t>>;
using ShardingArrayRef = ArrayRef<SmallVector<int64_t>>;

struct ShardingOption {
  ShardingArray shardingArray;
  SymbolRefAttr cluster;
  ShardingOption() = default;
  ShardingOption(const ShardingArray &shardingArray, SymbolRefAttr cluster)
      : shardingArray(shardingArray), cluster(cluster) {}
};

constexpr StringRef getShardingArrayName() { return "sharding_array"; }

constexpr StringRef getMeshClusterName() { return "mesh_cluster"; }

namespace detail {

FailureOr<ShardingOption> defaultGetShardingOption(Operation *op, OpBuilder &b);

LogicalResult defaultSetShardingAnnotations(
    Operation *op, const ShardingOption &shardingOption, OpBuilder &b);

} // namespace detail

} // namespace mesh

} // namespace mlir

/// Include the ODS generated interface header files.
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h.inc"

#endif // MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_