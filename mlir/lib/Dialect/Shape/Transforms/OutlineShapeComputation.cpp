//====----- OutlineShapeComputation.cpp -------------------------*- C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <unordered_set>
#include <vector>

#define DEBUG_TYPE "outline-shape-computation"

using namespace mlir;

namespace {

// A Value is an input of the cluster if it is an operand of an operation in the
// cluster and its defining operation is not in the cluster.
SmallVector<Value, 4>
getInputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster) {
  SmallVector<Value, 4> inputs;
  llvm::SmallDenseSet<Value> inputSet;
  llvm::SmallDenseSet<Operation *> opSet;
  for (Operation *op : cluster) {
    bool inserted = opSet.insert(op).second;
    (void)inserted;
    assert(inserted && "cluster contains duplicate operations");
  }

  for (Operation *op : cluster) {
    for (Value operand : op->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      if (opSet.find(operandOp) != opSet.end()) {
        // skip if defining op is in the cluster
        continue;
      }
      if (inputSet.insert(operand).second) {
        inputs.push_back(operand);
      }
    }
  }
  return inputs;
}

// Create a shape.func representing the shape computation for \p dimSize.
std::pair<shape::FuncOp, SmallVector<Value>>
createFuncFromCluster(OpBuilder &b, const SmallVector<Operation *, 8> &cluster,
                      Value dimSize, StringRef fnName) {
  if (cluster.size() == 0) {
    llvm_unreachable("There must be at least one element in the cluster.");
  }

  SmallVector<Value, 4> inputs = getInputsOfCluster(cluster);
  SmallVector<Type, 1> outputTypes{dimSize.getType()};
  SmallVector<Type, 4> inputTypes = llvm::to_vector(
      llvm::map_range(inputs, [](Value inp) { return inp.getType(); }));

  auto fnType = b.getFunctionType(inputTypes, outputTypes);
  b.setInsertionPointAfter(cluster[0]->getParentOp());
  shape::FuncOp fnOp =
      b.create<shape::FuncOp>(UnknownLoc::get(b.getContext()), fnName, fnType);
  Block *block = fnOp.addEntryBlock();
  b.setInsertionPoint(block, block->end());
  BlockAndValueMapping bvm;
  for (auto inputAndArg : llvm::zip(inputs, fnOp.getArguments())) {
    bvm.map(std::get<0>(inputAndArg), std::get<1>(inputAndArg));
  }
  for (Operation *op : cluster) {
    b.clone(*op, bvm);
  }
  llvm::SmallVector<Value, 4> fnReturns;
  fnReturns.push_back(bvm.lookupOrDefault(dimSize));

  b.create<shape::ReturnOp>(UnknownLoc::get(b.getContext()), fnReturns);
  fnOp.setPrivate();
  return std::make_pair(fnOp, inputs);
}

// The operations in the cluster might be unsorted, which could be inconvinient
// when creating shape.func op.
DenseMap<Value, SmallVector<Operation *, 8>>
getOrderedClusters(const DenseMap<Value, DenseSet<Operation *>> &clusters,
                   func::FuncOp funcOp) {
  DenseMap<Operation *, SmallVector<Value>> op2Shapes;
  for (auto it : clusters) {
    Value shape = it.first;
    const DenseSet<Operation *> &cluster = it.second;
    for (Operation *cOp : cluster) {
      op2Shapes[cOp].push_back(shape);
    }
  }

  DenseMap<Value, SmallVector<Operation *, 8>> orderedClusters;
  funcOp.walk([&](Operation *op) {
    auto it = op2Shapes.find(op);
    if (it != op2Shapes.end()) {
      Operation *cOp = it->first;
      for (Value shape : it->second) {
        orderedClusters[shape].push_back(cOp);
      }
    }
  });

  return orderedClusters;
}

void initFuncArgs(func::FuncOp funcOp,
                  std::unordered_set<std::string> &usedSymbolNames) {
  for (Value arg : funcOp.getArguments()) {
    auto rankedType = arg.getType().dyn_cast<RankedTensorType>();
    if (rankedType != nullptr) {
      auto encoding = rankedType.getEncoding().dyn_cast_or_null<ArrayAttr>();
      if (encoding != nullptr) {
        SmallVector<FlatSymbolRefAttr> symbols;
        for (Attribute attr : encoding) {
          if (auto symbolAttr = attr.dyn_cast<FlatSymbolRefAttr>()) {
            usedSymbolNames.insert(symbolAttr.getValue().str());
            symbols.push_back(symbolAttr);
          }
        }

      //   for (Operation *user : arg.getUsers()) {
      //     if (llvm::isa<shape::ShapeOfOp>(user)) {
      //       for (Operation *extentUser : user->getUsers()) {
      //         if (auto extentOp =
      //                 llvm::dyn_cast<shape::GetExtentOp>(extentUser)) {
      //           APInt dimPos;
      //           if (!matchPattern(extentOp.getDim(), m_ConstantInt(&dimPos))) {
      //             llvm_unreachable(
      //                 "The dim value of shape.get_extent is not constant-like");
      //           }
      //         }
      //       }
      //     }
      //   }
      // }
    }
  }
}

// Increment \p idx until find the next available symbol name
std::string
getNextAvailableSymbolName(const std::string &prefix, int &idx,
                           std::unordered_set<std::string> &usedSymbolNames) {
  std::string name = prefix + std::to_string(idx++);

  while (usedSymbolNames.count(name)) {
    name = prefix + std::to_string(idx++);
  }
  usedSymbolNames.insert(name);
  return name;
}

void constructShapeFunc(const std::vector<shape::WithOp> &allWithOps,
                        MLIRContext *context,
                        std::unordered_set<std::string> &usedSymbolNames,
                        DenseMap<Value, SmallVector<Operation *, 8>> &clusters,
                        SymbolTable &symbolTable) {
  DenseMap<Value, FlatSymbolRefAttr> dynDimSrc2Symbol;
  std::string dynamicSourceNamePrefix = "s";
  int dynamicSourceNameIdx = 0;
  std::string shapeCalculationNamePrefix = "shape_cal_";
  int shapeCalculationNameIdx = 0;
  OpBuilder builder(context);

  auto getOrConstructSymbolFromDimSize = [&](Value dimSize) {
    auto symbolIt = dynDimSrc2Symbol.find(dimSize);
    if (symbolIt == dynDimSrc2Symbol.end()) {
      // FIXME: handle func arg
      std::string name = getNextAvailableSymbolName(
          dynamicSourceNamePrefix, dynamicSourceNameIdx, usedSymbolNames);
      auto symbol = FlatSymbolRefAttr::get(context, name);
      dynDimSrc2Symbol[dimSize] = symbol;
      return symbol;
    } else {
      return symbolIt->second;
    }
  };

  // Construct a shape function or a symbol for each cluster
  for (shape::WithOp withOp : allWithOps) {
    Value value = withOp.getOperand();
    Value shape = withOp.getShape();
    RankedTensorType rankedType = value.getType().dyn_cast<RankedTensorType>();
    if (rankedType == nullptr) {
      continue;
    }
    SmallVector<FlatSymbolRefAttr> symbols;

    // FIXME: handle the case that shape is not constructed from
    // shape.from_extents
    SmallVector<Attribute> arrayAttr;
    if (auto fromExtentsOp = shape.getDefiningOp<shape::FromExtentsOp>()) {
      for (auto it : llvm::enumerate(fromExtentsOp.getExtents())) {
        size_t idx = it.index();
        if (!rankedType.isDynamicDim(idx))
          continue;
        SmallVector<Attribute> symbols;
        Value dimSize = it.value();
        const SmallVector<Operation *, 8> &cluster = clusters[dimSize];
        // The cluster is empty if the dimension is from internal dynamic
        // dimension source or a function argument
        if (cluster.empty()) {
          FlatSymbolRefAttr symbol = getOrConstructSymbolFromDimSize(dimSize);
          symbols.push_back(symbol);
          LLVM_DEBUG(llvm::dbgs()
                     << "Symbol for " << dimSize << ": " << symbol << "\n");
        } else {
          std::string name = getNextAvailableSymbolName(
              shapeCalculationNamePrefix, shapeCalculationNameIdx,
              usedSymbolNames);
          auto pair = createFuncFromCluster(builder, cluster, dimSize, name);
          shape::FuncOp shapeFuncOp = pair.first;
          StringAttr insertedName = symbolTable.insert(shapeFuncOp);
          auto symbol = FlatSymbolRefAttr::get(context, insertedName);
          const SmallVector<Value> &inputs = pair.second;
          symbols.push_back(symbol);
          for (Value inp : inputs) {
            FlatSymbolRefAttr argSymbol = getOrConstructSymbolFromDimSize(inp);
            symbols.push_back(argSymbol);
          }
          arrayAttr.push_back(ArrayAttr::get(context, symbols));
          LLVM_DEBUG(llvm::dbgs()
                     << "Symbol for " << dimSize << ": " << arrayAttr.back());
        }
      }
    }

    value.setType(RankedTensorType::get(rankedType.getShape(),
                                        rankedType.getElementType(),
                                        ArrayAttr::get(context, arrayAttr)));
  }
}

struct OutlineShapeComputationPass
    : public OutlineShapeComputationBase<OutlineShapeComputationPass> {

  OutlineShapeComputationPass(const std::string &entryFunc)
      : OutlineShapeComputationBase() {
    this->entryFunc = entryFunc;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    MLIRContext *context = moduleOp.getContext();

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.getName() != entryFunc)
        return;

      // initialize class member \p onlyUsedByWithShapes_
      funcOp.walk(
          [&](Operation *op) { calOnlyUsedByWithShapesRecursively(op); });

      // collect all the shape.with_shape ops.
      std::vector<shape::WithOp> allWithOps;
      funcOp.walk([&](shape::WithOp withOp) { allWithOps.push_back(withOp); });

      std::unordered_set<std::string> usedSymbolNames;
      DenseMap<Value, FlatSymbolRefAttr> dynDimSrc2Symbol;
      initFuncArgs(funcOp, usedSymbolNames);

      DenseMap<Value, SmallVector<Operation *, 8>> clusters =
          constructClustersForEachDimSize(allWithOps, funcOp);

      // Here we assume shape.with_shape ops has no users
      for (shape::WithOp withOp : allWithOps) {
        if (withOp.use_empty())
          withOp->erase();
      }

      // dce
      if (failed(applyPatternsAndFoldGreedily(funcOp, {}))) {
        return signalPassFailure();
      }
    });

    // // FIXME: fix nested call
    // moduleOp.walk([&](func::FuncOp funcOp) {
    //   funcOp.setType(
    //       FunctionType::get(context, funcOp.front().getArgumentTypes(),
    //                         funcOp.front().getTerminator()->getOperandTypes()));
    // });
  }

private:
  bool calOnlyUsedByWithShapesRecursively(Operation *op);
  void getClusterFromValue(Value shape,
                           DenseMap<Value, DenseSet<Operation *>> &clusters);
  DenseMap<Value, SmallVector<Operation *, 8>>
  constructClustersForEachDimSize(const std::vector<shape::WithOp> &allWithOps,
                                  func::FuncOp funcOp);
  DenseMap<Operation *, bool> onlyUsedByWithShapes_;
};

DenseMap<Value, SmallVector<Operation *, 8>>
OutlineShapeComputationPass::constructClustersForEachDimSize(
    const std::vector<shape::WithOp> &allWithOps, func::FuncOp funcOp) {
  DenseMap<Value, DenseSet<Operation *>> clusters;
  for (shape::WithOp withOp : allWithOps) {
    Value shape = withOp.getShape();
    // Get the operations cluster for each dimension size
    // FIXME: handle the case that shape is not constructed from
    // shape.from_extents
    if (auto fromExtentsOp = shape.getDefiningOp<shape::FromExtentsOp>()) {
      for (Value dimSize : fromExtentsOp.getExtents()) {
        if (clusters.count(dimSize) == 0)
          getClusterFromValue(dimSize, clusters);
      }
    }
  }
  return getOrderedClusters(clusters, funcOp);
}

void OutlineShapeComputationPass::getClusterFromValue(
    Value shape, DenseMap<Value, DenseSet<Operation *>> &clusters) {
  DenseSet<Operation *> cluster;

  Operation *defOp = shape.getDefiningOp();
  if (nullptr == defOp) {
    return;
  }

  DenseSet<Operation *> visited;
  std::queue<Operation *> queue;
  visited.insert(defOp);
  queue.push(defOp);
  while (!queue.empty()) {
    Operation *op = queue.front();
    queue.pop();
    if (op->getNumOperands() == 0) {
      cluster.insert(op);
    } else if (llvm::isa<tensor::DimOp>(op) &&
               !onlyUsedByWithShapes_.count(
                   op->getOperand(0).getDefiningOp())) {
      continue;
    } else {
      cluster.insert(op);
      for (Value inp : op->getOperands()) {
        Operation *inpDefOp = inp.getDefiningOp();
        if (nullptr != inpDefOp && !visited.contains(inpDefOp)) {
          visited.insert(inpDefOp);
          queue.push(inpDefOp);
        }
      }
    }
  }

  clusters[shape] = std::move(cluster);
}

// check if an operation is only used by shape.with_shape directly or
// indirectly.
bool OutlineShapeComputationPass::calOnlyUsedByWithShapesRecursively(
    Operation *op) {
  auto it = onlyUsedByWithShapes_.find(op);
  if (it != onlyUsedByWithShapes_.end())
    return it->second;

  if (llvm::isa<shape::WithOp>(op)) {
    onlyUsedByWithShapes_[op] = true;
    return true;
  }

  if (op->use_empty()) {
    onlyUsedByWithShapes_[op] = false;
    return false;
  }

  bool allUsers = true;
  for (Operation *op : op->getUsers()) {
    allUsers |= calOnlyUsedByWithShapesRecursively(op);
  }

  onlyUsedByWithShapes_[op] = allUsers;
  return allUsers;
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createOutlineShapeComputationPass(const std::string &entryFunc) {
  return std::make_unique<OutlineShapeComputationPass>(entryFunc);
}
