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
#include "mlir/Transforms/DialectConversion.h"
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

// Create a shape.func representing the shape computation for \p shape.
std::pair<shape::FuncOp, SmallVector<Value>>
createFuncFromCluster(OpBuilder &b, const SmallVector<Operation *, 8> &cluster,
                      Value shape, StringRef fnName) {
  if (cluster.size() == 0) {
    llvm_unreachable("There must be at least one element in the cluster.");
  }

  SmallVector<Value, 4> inputs = getInputsOfCluster(cluster);
  SmallVector<Type, 1> outputTypes{shape.getType()};
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
  fnReturns.push_back(bvm.lookupOrDefault(shape));

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

// return argument index if \p shape is the output of a
// shape.shape_of(func_arg), else return -1.
int getShapeOfFuncArgIdx(Value shape, func::FuncOp funcOp) {
  shape::ShapeOfOp shapeOfOp = shape.getDefiningOp<shape::ShapeOfOp>();
  if (shapeOfOp == nullptr)
    return false;
  Value inp = shapeOfOp.getArg();
  for (int i = 0; i < int(funcOp.getNumArguments()); ++i) {
    if (funcOp.getArgument(i) == inp) {
      return i;
    }
  }

  return -1;
}

void constructShapeFunc(const std::vector<shape::WithOp> &allWithOps,
                        MLIRContext *context,
                        DenseMap<Value, SmallVector<Operation *, 8>> &clusters,
                        SymbolTable &symbolTable, func::FuncOp funcOp) {
  std::unordered_set<std::string> usedSymbolNames;
  DenseMap<Value, FlatSymbolRefAttr> dynShapeSrc2Symbol;
  std::string dynamicSourceNamePrefix = "s";
  int dynamicSourceNameIdx = 0;
  std::string shapeCalculationNamePrefix = "shape_cal_";
  int shapeCalculationNameIdx = 0;
  OpBuilder builder(context);

  auto getOrConstructSymbolFromShape = [&](Value shape) {
    auto symbolIt = dynShapeSrc2Symbol.find(shape);
    if (symbolIt == dynShapeSrc2Symbol.end()) {
      std::string name;
      int index = getShapeOfFuncArgIdx(shape, funcOp);
      if (index >= 0) {
        name = "arg_" + std::to_string(index);
      } else {
        name = getNextAvailableSymbolName(
            dynamicSourceNamePrefix, dynamicSourceNameIdx, usedSymbolNames);
      }
      auto symbol = FlatSymbolRefAttr::get(context, name);
      dynShapeSrc2Symbol[shape] = symbol;
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
    const SmallVector<Operation *, 8> &cluster = clusters[shape];
    // The cluster is empty when the shape is equal to a dynamic shape source
    if (cluster.empty()) {
      FlatSymbolRefAttr symbol = getOrConstructSymbolFromShape(shape);
      value.setType(RankedTensorType::get(rankedType.getShape(),
                                          rankedType.getElementType(), symbol));
      LLVM_DEBUG(llvm::dbgs()
                 << "Symbol for " << shape << ": " << symbol << "\n");
    } else {
      SmallVector<Attribute> symbols;
      std::string name = getNextAvailableSymbolName(
          shapeCalculationNamePrefix, shapeCalculationNameIdx, usedSymbolNames);
      auto pair = createFuncFromCluster(builder, cluster, shape, name);
      const SmallVector<Value> &inputs = pair.second;
      shape::FuncOp shapeFuncOp = pair.first;
      StringAttr insertedName = symbolTable.insert(shapeFuncOp);
      auto symbol = FlatSymbolRefAttr::get(context, insertedName);
      symbols.push_back(symbol);
      for (Value inp : inputs) {
        FlatSymbolRefAttr argSymbol = getOrConstructSymbolFromShape(inp);
        symbols.push_back(argSymbol);
      }
      auto arrayAttr = ArrayAttr::get(context, symbols);
      LLVM_DEBUG(llvm::dbgs()
                 << "Symbol for " << shape << ": " << arrayAttr << "\n");
      value.setType(RankedTensorType::get(
          rankedType.getShape(), rankedType.getElementType(), arrayAttr));
    }
  }
}

struct OutlineShapeComputationPass
    : public OutlineShapeComputationBase<OutlineShapeComputationPass> {

  OutlineShapeComputationPass(const std::string &entryFunc)
      : OutlineShapeComputationBase() {
    this->entryFunc = entryFunc;
  }

  void runOnOperation() override;

private:
  bool calOnlyUsedByWithShapesRecursively(Operation *op);
  void getClusterFromValue(Value shape,
                           DenseMap<Value, DenseSet<Operation *>> &clusters);
  DenseMap<Value, SmallVector<Operation *, 8>>
  constructClustersForEachShape(const std::vector<shape::WithOp> &allWithOps,
                                func::FuncOp funcOp);
  DenseMap<Operation *, bool> onlyUsedByWithShapes_;
};

class TensorDimOpRewriter : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    auto shapeOf =
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), op.getSource());
    rewriter.replaceOpWithNewOp<shape::GetExtentOp>(op, op.getType(), shapeOf,
                                                    op.getIndex());
    return success();
  }
};

void OutlineShapeComputationPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  SymbolTable symbolTable(moduleOp);

  moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp.getName() != entryFunc)
      return;

    MLIRContext *context = funcOp.getContext();
    RewritePatternSet prevPatterns(context);
    prevPatterns.insert<TensorDimOpRewriter>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(prevPatterns)))) {
      return signalPassFailure();
    }

    // initialize class member \p onlyUsedByWithShapes_
    onlyUsedByWithShapes_.clear();
    funcOp.walk([&](Operation *op) { calOnlyUsedByWithShapesRecursively(op); });

    // collect all the shape.with_shape ops.
    std::vector<shape::WithOp> allWithOps;
    funcOp.walk([&](shape::WithOp withOp) { allWithOps.push_back(withOp); });

    DenseMap<Value, SmallVector<Operation *, 8>> clusters =
        constructClustersForEachShape(allWithOps, funcOp);
    constructShapeFunc(allWithOps, context, clusters, symbolTable, funcOp);

    for (shape::WithOp withOp : allWithOps) {
      Value value = withOp.getOperand();
      for (Operation *user : withOp.getResult().getUsers()) {
        if (Value valueOf = llvm::dyn_cast<shape::ValueOfOp>(user)) {
          valueOf.replaceAllUsesExcept(value, withOp);
        }
      }
    }

    // dce
    if (failed(applyPatternsAndFoldGreedily(funcOp, {}))) {
      return signalPassFailure();
    }

    funcOp.setType(
        FunctionType::get(context, funcOp.front().getArgumentTypes(),
                          funcOp.front().getTerminator()->getOperandTypes()));
  });
}

DenseMap<Value, SmallVector<Operation *, 8>>
OutlineShapeComputationPass::constructClustersForEachShape(
    const std::vector<shape::WithOp> &allWithOps, func::FuncOp funcOp) {
  DenseMap<Value, DenseSet<Operation *>> clusters;
  for (shape::WithOp withOp : allWithOps) {
    Value shape = withOp.getShape();
    if (clusters.count(shape) == 0) {
      getClusterFromValue(shape, clusters);
    }
  }
  return getOrderedClusters(clusters, funcOp);
}

// The output of a cluster is the \p shape, and the inputs are either the result
// of shape.shape_of or function argument.
void OutlineShapeComputationPass::getClusterFromValue(
    Value shape, DenseMap<Value, DenseSet<Operation *>> &clusters) {
  DenseSet<Operation *> cluster;

  Operation *defOp = shape.getDefiningOp();
  // defOp == nullptr means shape is the argument of the func op
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
    } else if (llvm::isa<shape::ShapeOfOp>(op) &&
               !onlyUsedByWithShapes_.count(
                   op->getOperand(0).getDefiningOp())) {
      // Stop when the op is type of shape.shape_of and its operand isn't only
      // used by shape.with_shape ops
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
