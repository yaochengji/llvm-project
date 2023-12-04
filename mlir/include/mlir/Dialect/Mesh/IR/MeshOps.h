//===- MeshOps.h - Mesh Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_IR_MESHOPS_H
#define MLIR_DIALECT_MESH_IR_MESHOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>

#include "mlir/Dialect/Mesh/IR/MeshOpsDialect.h.inc"

#include "mlir/Dialect/Mesh/IR/MeshOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOps.h.inc"

namespace mlir {

class SymbolTableCollection;

namespace mesh {

struct DimensionSize {
  static DimensionSize dynamic() { return DimensionSize(ShapedType::kDynamic); }
  DimensionSize(int64_t val) : val(val) {}
  int64_t value() const { return val; }
  operator int64_t() const { return val; }
  bool isDynamic() const { return ShapedType::isDynamic(val); }

private:
  int64_t val;
};

DimensionSize operator/(DimensionSize lhs, DimensionSize rhs);

DimensionSize operator*(DimensionSize lhs, DimensionSize rhs);

bool isReductionLoop(IteratorType iType);

bool areReductionAndPartialMatch(IteratorType iType, Partial partial);

template <typename T>
void removeTrailingEmptySubArray(SmallVector<SmallVector<T>> &array) {
  while (!array.empty() && array.back().empty())
    array.pop_back();
}

Partial getPartialTypeFromReduction(IteratorType iType);

FailureOr<ClusterOp> getMesh(Operation *op, FlatSymbolRefAttr meshSymbol,
                             SymbolTableCollection &symbolTable);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_IR_MESHOPS_H
