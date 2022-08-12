// RUN: mlir-opt -allow-unregistered-dialect -outline-shape-computation="entry-func=main" %s | FileCheck %s


func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) -> tensor<?x4x?xf32> {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = shape.shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
  %1 = shape.get_extent %0, %c2 : tensor<3xindex>, index -> index
  %2 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
  %3 = shape.get_extent %0, %c0 : tensor<3xindex>, index -> index
  %4 = arith.addi %3, %c2 : index
  %5 = shape.from_extents %4, %c4, %1 : index, index, index
  %6 = shape.with_shape %2, %5 : tensor<?x4x?xf32>, !shape.shape
  %7 = shape.value_of %6 : !shape.value_shape -> tensor<?x4x?xf32>
  return %7 : tensor<?x4x?xf32>
}

// CHECK-LABEL: func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) -> tensor<?x4x?xf32, [@shape_cal_0, @arg_0]> {
// CHECK-NEXT:     %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32, [@shape_cal_0, @arg_0]>
// CHECK-NEXT:     return %0 : tensor<?x4x?xf32, [@shape_cal_0, @arg_0]>
  
// CHECK-LABEL:  shape.func private @shape_cal_0(%arg0: tensor<3xindex>) -> !shape.shape {
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     %0 = get_extent %arg0, %c2 : tensor<3xindex>, index -> index
// CHECK-NEXT:     %1 = get_extent %arg0, %c0 : tensor<3xindex>, index -> index
// CHECK-NEXT:     %2 = arith.addi %1, %c2 : index
// CHECK-NEXT:     %3 = from_extents %2, %c4, %0 : index, index, index
// CHECK-NEXT:     return %3 : !shape.shape
