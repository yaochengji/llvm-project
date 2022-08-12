// RUN: mlir-opt -insert-dimension-symbols="entry-func=main" -allow-unregistered-dialect -split-input-file %s | FileCheck %s


func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x?x?xf32>) -> tensor<?x4x?xf32> {
  %0 =  "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<?x4x?xf32>, tensor<2x?x?xf32>) -> tensor<?x4x?xf32>
  return %0 : tensor<?x4x?xf32>
}
// CHECK: @main(%arg0: tensor<?x4x?xf32, [@s0, @s1]>, %arg1: tensor<2x?x?xf32, [@s2, @s3]>)