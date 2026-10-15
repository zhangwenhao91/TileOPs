module {
  func.func @vadd_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32) attributes {BlockIdx = 3 : i64} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32, 256], strides: [256, 1] : memref<?xf32> to memref<32x256xf32, strided<[256, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32, 256], strides: [256, 1] : memref<?xf32> to memref<32x256xf32, strided<[256, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [32, 256], strides: [256, 1] : memref<?xf32> to memref<32x256xf32, strided<[256, 1]>>
    %alloc = memref.alloc() : memref<16384xi8, 1>
    %c0 = arith.constant 0 : index
    %view = memref.view %alloc[%c0][] : memref<16384xi8, 1> to memref<16x256xf32, 1>
    %alloc_2 = memref.alloc() : memref<16384xi8, 1>
    %c0_3 = arith.constant 0 : index
    %view_4 = memref.view %alloc_2[%c0_3][] : memref<16384xi8, 1> to memref<16x256xf32, 1>
    %alloc_5 = memref.alloc() : memref<16384xi8, 1>
    %c0_6 = arith.constant 0 : index
    %view_7 = memref.view %alloc_5[%c0_6][] : memref<16384xi8, 1> to memref<16x256xf32, 1>
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %c0_i32 : i32 to index
    %c16_i32 = arith.constant 16 : i32
    %1 = arith.muli %arg3, %c16_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %c256 = arith.constant 256 : index
    %3 = arith.muli %2, %c256 : index
    %4 = arith.addi %0, %3 : index
    %c0_i32_8 = arith.constant 0 : i32
    %5 = arith.index_cast %c0_i32_8 : i32 to index
    %c1 = arith.constant 1 : index
    %6 = arith.muli %5, %c1 : index
    %7 = arith.addi %4, %6 : index
    %reinterpret_cast_9 = memref.reinterpret_cast %reinterpret_cast to offset: [%7], sizes: [16, 256], strides: [256, 1] : memref<32x256xf32, strided<[256, 1]>> to memref<16x256xf32, strided<[256, 1], offset: ?>>
    "tilelang.copy"(%reinterpret_cast_9, %view) : (memref<16x256xf32, strided<[256, 1], offset: ?>>, memref<16x256xf32, 1>) -> ()
    %c0_i32_10 = arith.constant 0 : i32
    %8 = arith.index_cast %c0_i32_10 : i32 to index
    %c16_i32_11 = arith.constant 16 : i32
    %9 = arith.muli %arg3, %c16_i32_11 : i32
    %10 = arith.index_cast %9 : i32 to index
    %c256_12 = arith.constant 256 : index
    %11 = arith.muli %10, %c256_12 : index
    %12 = arith.addi %8, %11 : index
    %c0_i32_13 = arith.constant 0 : i32
    %13 = arith.index_cast %c0_i32_13 : i32 to index
    %c1_14 = arith.constant 1 : index
    %14 = arith.muli %13, %c1_14 : index
    %15 = arith.addi %12, %14 : index
    %reinterpret_cast_15 = memref.reinterpret_cast %reinterpret_cast_0 to offset: [%15], sizes: [16, 256], strides: [256, 1] : memref<32x256xf32, strided<[256, 1]>> to memref<16x256xf32, strided<[256, 1], offset: ?>>
    "tilelang.copy"(%reinterpret_cast_15, %view_4) : (memref<16x256xf32, strided<[256, 1], offset: ?>>, memref<16x256xf32, 1>) -> ()
    "tilelang.scope"() ({
      %alloc_22 = memref.alloc() : memref<16x256xf32, 2>
      %alloc_23 = memref.alloc() : memref<16x256xf32, 2>
      %alloc_24 = memref.alloc() : memref<16x256xf32, 2>
      "tilelang.copy"(%view, %alloc_22) : (memref<16x256xf32, 1>, memref<16x256xf32, 2>) -> ()
      "tilelang.copy"(%view_4, %alloc_23) : (memref<16x256xf32, 1>, memref<16x256xf32, 2>) -> ()
      linalg.elementwise <add> ins(%alloc_22, %alloc_23 : memref<16x256xf32, 2>, memref<16x256xf32, 2>) outs(%alloc_24 : memref<16x256xf32, 2>)
      "tilelang.copy"(%alloc_24, %view_7) : (memref<16x256xf32, 2>, memref<16x256xf32, 1>) -> ()
    }) {mode = #tilelang.scope_mode<simd>} : () -> ()
    %c0_i32_16 = arith.constant 0 : i32
    %16 = arith.index_cast %c0_i32_16 : i32 to index
    %c16_i32_17 = arith.constant 16 : i32
    %17 = arith.muli %arg3, %c16_i32_17 : i32
    %18 = arith.index_cast %17 : i32 to index
    %c256_18 = arith.constant 256 : index
    %19 = arith.muli %18, %c256_18 : index
    %20 = arith.addi %16, %19 : index
    %c0_i32_19 = arith.constant 0 : i32
    %21 = arith.index_cast %c0_i32_19 : i32 to index
    %c1_20 = arith.constant 1 : index
    %22 = arith.muli %21, %c1_20 : index
    %23 = arith.addi %20, %22 : index
    %reinterpret_cast_21 = memref.reinterpret_cast %reinterpret_cast_1 to offset: [%23], sizes: [16, 256], strides: [256, 1] : memref<32x256xf32, strided<[256, 1]>> to memref<16x256xf32, strided<[256, 1], offset: ?>>
    "tilelang.copy"(%view_7, %reinterpret_cast_21) : (memref<16x256xf32, 1>, memref<16x256xf32, strided<[256, 1], offset: ?>>) -> ()
    return
  }
}
