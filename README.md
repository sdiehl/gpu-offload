# GPU Offload Tutorial

This minimal project demonstrates how to compile MLIR code to PTX and execute it on NVIDIA GPUs using Python.

- `compile.py`: Functions for compiling MLIR to PTX
- `run.py`: Functions for running PTX kernels on CUDA GPUs

## Installation

```bash
poetry install
poetry install --with gpu
```

## Usage

```python
import numpy as np
from compile import compile_mlir_to_ptx
from run import setup_cuda, run_ptx_kernel, cleanup_cuda

# Define MLIR module
mlir_code = """
module {
  func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %x0 : tensor<10x10xf32>
  }
}
"""

# Compile MLIR to PTX
ptx_code = compile_mlir_to_ptx(mlir_code)

# Setup CUDA
context = setup_cuda()

# Prepare data
input_data = np.random.randn(10, 10).astype(np.float32)
output_data = np.zeros((10, 10), dtype=np.float32)

# Run kernel
run_ptx_kernel(
    ptx_code,
    "square_kernel",
    ["ptr:in", "ptr:out"],
    input_data,
    output_data,
    grid_dim=(10,),
    block_dim=(16,)
)

# Clean up
cleanup_cuda(context)
```

## Running the Example

```bash
poetry run python example.py
poetry run python mlir_example.py
poetry run python ptx_example.py
```


## Pipeline

Start MLIR module.

```mlir
module {
  func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    linalg.square ins(%arg0 : memref<10x10xf32>) outs(%arg1 : memref<10x10xf32>)
    return %arg1 : memref<10x10xf32>
  }
}
```

After `one-shot-bufferize` pass.

```mlir
module {
  func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    affine.for %arg2 = 0 to 10 {
      affine.for %arg3 = 0 to 10 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
        %1 = arith.mulf %0, %0 : f32
        affine.store %1, %arg1[%arg2, %arg3] : memref<10x10xf32>
      }
    }
    return %arg1 : memref<10x10xf32>
  }
}
```

After `convert-affine-for-to-gpu` pass.

```mlir
func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = arith.subi %c10, %c0 : index
  %c1 = arith.constant 1 : index
  %c0_0 = arith.constant 0 : index
  %c10_1 = arith.constant 10 : index
  %1 = arith.subi %c10_1, %c0_0 : index
  %c1_2 = arith.constant 1 : index
  %c1_3 = arith.constant 1 : index
  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %0, %arg9 = %c1_3, %arg10 = %c1_3) threads(%arg5, %arg6, %arg7) in (%arg11 = %1, %arg12 = %c1_3, %arg13 = %c1_3) {
    %2 = arith.addi %c0, %arg2 : index
    %3 = arith.addi %c0_0, %arg5 : index
    %4 = affine.load %arg0[%2, %3] : memref<10x10xf32>
    %5 = arith.mulf %4, %4 : f32
    affine.store %5, %arg1[%2, %3] : memref<10x10xf32>
    gpu.terminator
  }
  return %arg1 : memref<10x10xf32>
}
```

After `gpu-kernel-outlining` pass.

```mlir
module attributes {gpu.container_module} {
  func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %0 = arith.subi %c10, %c0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c10_1 = arith.constant 10 : index
    %1 = arith.subi %c10_1, %c0_0 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    gpu.launch_func  @square_kernel::@square_kernel blocks in (%0, %c1_3, %c1_3) threads in (%1, %c1_3, %c1_3)  args(%c0 : index, %c0_0 : index, %arg0 : memref<10x10xf32>, %arg1 : memref<10x10xf32>)
    return %arg1 : memref<10x10xf32>
  }
  gpu.module @square_kernel {
    gpu.func @square_kernel(%arg0: index, %arg1: index, %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>) kernel {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = arith.addi %arg0, %block_id_x : index
      %1 = arith.addi %arg1, %thread_id_x : index
      %2 = affine.load %arg2[%0, %1] : memref<10x10xf32>
      %3 = arith.mulf %2, %2 : f32
      affine.store %3, %arg3[%0, %1] : memref<10x10xf32>
      gpu.return
    }
  }
}
```

After `lower-affine` pass.

```mlir
module attributes {gpu.container_module} {
  func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %0 = arith.subi %c10, %c0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c10_1 = arith.constant 10 : index
    %1 = arith.subi %c10_1, %c0_0 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    gpu.launch_func  @square_kernel::@square_kernel blocks in (%0, %c1_3, %c1_3) threads in (%1, %c1_3, %c1_3)  args(%c0 : index, %c0_0 : index, %arg0 : memref<10x10xf32>, %arg1 : memref<10x10xf32>)
    return %arg1 : memref<10x10xf32>
  }
  gpu.module @square_kernel {
    gpu.func @square_kernel(%arg0: index, %arg1: index, %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>) kernel {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = arith.addi %arg0, %block_id_x : index
      %1 = arith.addi %arg1, %thread_id_x : index
      %2 = memref.load %arg2[%0, %1] : memref<10x10xf32>
      %3 = arith.mulf %2, %2 : f32
      memref.store %3, %arg3[%0, %1] : memref<10x10xf32>
      gpu.return
    }
  }
}
```

After `gpu-decompose-memrefs` pass.

```mlir
module attributes {gpu.container_module} {
  func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    gpu.launch_func  @square_kernel::@square_kernel blocks in (%c10, %c1, %c1) threads in (%c10, %c1, %c1)  args(%c0 : index, %c0 : index, %arg0 : memref<10x10xf32>, %arg1 : memref<10x10xf32>)
    return %arg1 : memref<10x10xf32>
  }
  gpu.module @square_kernel {
    gpu.func @square_kernel(%arg0: index, %arg1: index, %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>) kernel {
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.addi %arg0, %block_id_x : index
      %1 = arith.addi %arg1, %thread_id_x : index
      %2 = memref.load %arg2[%0, %1] : memref<10x10xf32>
      %3 = arith.mulf %2, %2 : f32
      memref.store %3, %arg3[%0, %1] : memref<10x10xf32>
      gpu.return
    }
  }
}
```

After `convert-gpu-to-nvvm` pass.

```mlir
gpu.module @square_kernel {
  llvm.func @square_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, nvvm.kernel} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(10 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(10 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(10 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg2, %13[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg2, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(10 : index) : i64
    %19 = llvm.insertvalue %18, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(10 : index) : i64
    %21 = llvm.insertvalue %20, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(10 : index) : i64
    %23 = llvm.insertvalue %22, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.insertvalue %24, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = nvvm.read.ptx.sreg.ctaid.x : i32
    %27 = llvm.sext %26 : i32 to i64
    %28 = nvvm.read.ptx.sreg.tid.x : i32
    %29 = llvm.sext %28 : i32 to i64
    %30 = llvm.add %arg0, %27 : i64
    %31 = llvm.add %arg1, %29 : i64
    %32 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(10 : index) : i64
    %34 = llvm.mul %30, %33 : i64
    %35 = llvm.add %34, %31 : i64
    %36 = llvm.getelementptr %32[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %37 = llvm.load %36 : !llvm.ptr -> f32
    %38 = llvm.fmul %37, %37 : f32
    %39 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(10 : index) : i64
    %41 = llvm.mul %30, %40 : i64
    %42 = llvm.add %41, %31 : i64
    %43 = llvm.getelementptr %39[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %38, %43 : f32, !llvm.ptr
    llvm.return
  }
}
```

After `nvvm-attach-target` pass.

```mlir
module attributes {gpu.container_module} {
  func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    gpu.launch_func  @square_kernel::@square_kernel blocks in (%c10, %c1, %c1) threads in (%c10, %c1, %c1)  args(%c0 : index, %c0 : index, %arg0 : memref<10x10xf32>, %arg1 : memref<10x10xf32>)
    return %arg1 : memref<10x10xf32>
  }
  gpu.module @square_kernel [#nvvm.target<O = 3, chip = "sm_90", features = "+ptx80">] {
    llvm.func @square_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg3, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %3 = llvm.mlir.constant(0 : index) : i64
      %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %5 = llvm.mlir.constant(10 : index) : i64
      %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %7 = llvm.mlir.constant(10 : index) : i64
      %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %9 = llvm.mlir.constant(10 : index) : i64
      %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %11 = llvm.mlir.constant(1 : index) : i64
      %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %13 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg2, %13[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %15 = llvm.insertvalue %arg2, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %16 = llvm.mlir.constant(0 : index) : i64
      %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %18 = llvm.mlir.constant(10 : index) : i64
      %19 = llvm.insertvalue %18, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %20 = llvm.mlir.constant(10 : index) : i64
      %21 = llvm.insertvalue %20, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %22 = llvm.mlir.constant(10 : index) : i64
      %23 = llvm.insertvalue %22, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %24 = llvm.mlir.constant(1 : index) : i64
      %25 = llvm.insertvalue %24, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %26 = nvvm.read.ptx.sreg.ctaid.x : i32
      %27 = llvm.sext %26 : i32 to i64
      %28 = nvvm.read.ptx.sreg.tid.x : i32
      %29 = llvm.sext %28 : i32 to i64
      %30 = llvm.add %arg0, %27 : i64
      %31 = llvm.add %arg1, %29 : i64
      %32 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %33 = llvm.mlir.constant(10 : index) : i64
      %34 = llvm.mul %30, %33 : i64
      %35 = llvm.add %34, %31 : i64
      %36 = llvm.getelementptr %32[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %37 = llvm.load %36 : !llvm.ptr -> f32
      %38 = llvm.fmul %37, %37 : f32
      %39 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %40 = llvm.mlir.constant(10 : index) : i64
      %41 = llvm.mul %30, %40 : i64
      %42 = llvm.add %41, %31 : i64
      %43 = llvm.getelementptr %39[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %38, %43 : f32, !llvm.ptr
      llvm.return
    }
  }
}
```

```mlir
module attributes {gpu.container_module} {
  llvm.func @square(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(10 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(10 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(10 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg0, %13[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg0, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(10 : index) : i64
    %19 = llvm.insertvalue %18, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(10 : index) : i64
    %21 = llvm.insertvalue %20, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(10 : index) : i64
    %23 = llvm.insertvalue %22, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.insertvalue %24, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.mlir.constant(10 : index) : i64
    %29 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    gpu.launch_func  @square_kernel::@square_kernel blocks in (%28, %26, %26) threads in (%28, %26, %26) : i64 args(%27 : i64, %27 : i64, %29 : !llvm.ptr, %30 : !llvm.ptr)
    %31 = llvm.extractvalue %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.return %31 : !llvm.ptr
  }
  gpu.module @square_kernel [#nvvm.target<O = 3, chip = "sm_90", features = "+ptx80">] {
    llvm.func @square_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(10 : index) : i64
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = llvm.sext %1 : i32 to i64
      %3 = nvvm.read.ptx.sreg.tid.x : i32
      %4 = llvm.sext %3 : i32 to i64
      %5 = llvm.add %arg0, %2 : i64
      %6 = llvm.add %arg1, %4 : i64
      %7 = llvm.mul %5, %0 : i64
      %8 = llvm.add %7, %6 : i64
      %9 = llvm.getelementptr %arg2[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %10 = llvm.load %9 : !llvm.ptr -> f32
      %11 = llvm.fmul %10, %10 : f32
      %12 = llvm.mul %5, %0 : i64
      %13 = llvm.add %12, %6 : i64
      %14 = llvm.getelementptr %arg3[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %11, %14 : f32, !llvm.ptr
      llvm.return
    }
  }
}
```

After `llc -march=nvptx64 -mcpu=sm_90`.

```ptx
//
// Generated by LLVM NVPTX Back-End
//

.version 7.8
.target sm_90
.address_size 64

	// .globl	square_kernel           // -- Begin function square_kernel
                                        // @square_kernel
.visible .entry square_kernel(
	.param .u64 square_kernel_param_0,
	.param .u64 square_kernel_param_1,
	.param .u64 square_kernel_param_2,
	.param .u64 square_kernel_param_3
)
{
	.reg .b32 	%r<3>;
	.reg .f32 	%f<3>;
	.reg .b64 	%rd<16>;

// %bb.0:
	ld.param.u64 	%rd1, [square_kernel_param_0];
	ld.param.u64 	%rd2, [square_kernel_param_3];
	cvta.to.global.u64 	%rd3, %rd2;
	ld.param.u64 	%rd4, [square_kernel_param_1];
	ld.param.u64 	%rd5, [square_kernel_param_2];
	cvta.to.global.u64 	%rd6, %rd5;
	mov.u32 	%r1, %ctaid.x;
	cvt.s64.s32 	%rd7, %r1;
	mov.u32 	%r2, %tid.x;
	cvt.s64.s32 	%rd8, %r2;
	add.s64 	%rd9, %rd1, %rd7;
	add.s64 	%rd10, %rd4, %rd8;
	mul.lo.s64 	%rd11, %rd9, 10;
	add.s64 	%rd12, %rd11, %rd10;
	shl.b64 	%rd13, %rd12, 2;
	add.s64 	%rd14, %rd6, %rd13;
	ld.global.f32 	%f1, [%rd14];
	mul.rn.f32 	%f2, %f1, %f1;
	add.s64 	%rd15, %rd3, %rd13;
	st.global.f32 	[%rd15], %f2;
	ret;
                                        // -- End function
}

```
