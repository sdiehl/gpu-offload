# MLIR to PTX

This is a minimal demonstration of compiling MLIR code to PTX and executing it on an NVIDIA GPU using Python MLIR bindings and Python CUDA bindings.

- [`compile.py`](./compile.py): Functions for compiling MLIR to PTX
- [`run.py`](./run.py): Functions for running PTX kernels on GPU
- [`verify.py`](./verify.py): Verify the PTX code

To run the example in Google Colab, click the badge below. Launch an instance with a NVIDIA GPU like a T4 (`sm_75`) or A100 (`sm_80`).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdiehl/gpu-offload/blob/main/Minimal.ipynb)

Or load the following notebook in your local environment.

* [Minimal.ipynb](./Minimal.ipynb)
* [minimal.py](./minimal.py)

## Installation

If you don't want to use a notebook, see the [INSTALL.md](./INSTALL.md) file for a minimal installation. You will need a Linux system with a NVIDIA GPU, CUDA toolkit, and MLIR installed.

## Usage

```python
import numpy as np

from compile import compile_mlir_to_ptx
from run import CudaContext

# Example MLIR module for a matrix squaring operation
SQUARE_MLIR = """
module {
  func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %x0 : tensor<10x10xf32>
  }
}
"""

# Input data: 10x10 random matrix
size = 10
input_data = np.random.randn(size, size).astype(np.float32)

# Expected output for verification
expected_output = input_data * input_data

# Step 1: Compile MLIR to PTX
print("Compiling MLIR to PTX...")
ptx_code = compile_mlir_to_ptx(SQUARE_MLIR)

# Step 2: Execute the kernel using the CudaContext
with CudaContext() as ctx:
    print("Running kernel on GPU...")

    # Create device arrays
    d_input = ctx.array(input_data)
    d_output = ctx.array(shape=(size, size), dtype=np.float32)

    # Execute kernel
    # The kernel_name should match the name generated during compilation
    ctx.run_kernel(
        ptx_code,
        "square_kernel",
        [d_input, d_output],
        n=size * size,
        block_dims=(16, 1, 1),
    )

    # Get results back
    d_output.copy_device_to_host()
    result = d_output.host_array

    # Verify results
    np.testing.assert_allclose(result, expected_output, rtol=1e-5)
    print("Success! Results verified.")
```

## Running the Examples

```bash
poetry run python examples/example_mlir.py
poetry run python examples/example_ptx.py
poetry run python examples/example_full.py
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

After `one-shot-bufferize` pass. This pass uses `bufferize-function-boundaries` and `function-boundary-type-conversion=identity-layout-map` to properly handle tensor to buffer conversion at function boundaries while preserving layout information.

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

After `convert-affine-for-to-gpu` pass. This pass transforms affine loops into GPU kernel code with appropriate block and thread dimensions.

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

After `gpu-kernel-outlining` pass. This creates separate GPU modules and functions from the code inside GPU launch regions.

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

After `lower-affine` pass. This converts affine operations to standard operations to prepare for GPU-specific lowering.

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

After `gpu-decompose-memrefs` pass. This simplifies memref access patterns for GPU memory spaces.

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

After `convert-gpu-to-nvvm` pass. This uses `index-bitwidth=0` to use the default index size and `use-bare-ptr-memref-call-conv` to optimize memory access patterns with direct pointer manipulation.

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

After `nvvm-attach-target` pass. This configures the target GPU architecture with `chip=sm_90`, enables PTX 8.0 features with `features=+ptx80`, and sets optimization level to 3 with `O=3`. This targets the H100 (Hopper) architecture.

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

After `convert-gpu-to-nvvm` pass, this converts GPU operations to NVVM dialect operations. The subsequent `gpu-to-llvm` pass uses two key flags. The `use-bare-pointers-for-host` flag converts memref descriptors to raw pointers for host-side code. The `use-bare-pointers-for-kernels` flag converts memref descriptors to raw pointers for device kernel code.

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

After `mlir-translate --mlir-to-llvmir`. This step converts the MLIR LLVM dialect to standard LLVM IR format.

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @square_kernel(i64 %0, i64 %1, ptr %2, ptr %3) {
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = sext i32 %5 to i64
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = sext i32 %7 to i64
  %9 = add i64 %0, %6
  %10 = add i64 %1, %8
  %11 = mul i64 %9, 10
  %12 = add i64 %11, %10
  %13 = getelementptr float, ptr %2, i64 %12
  %14 = load float, ptr %13, align 4
  %15 = fmul float %14, %14
  %16 = mul i64 %9, 10
  %17 = add i64 %16, %10
  %18 = getelementptr float, ptr %3, i64 %17
  store float %15, ptr %18, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!nvvm.annotations = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @square_kernel, !"kernel", i32 1}
```

After `llc -march=nvptx64 -mcpu=sm_90`. This converts LLVM IR to PTX assembly code targeting the sm_90 architecture (H100 / Hopper).

```asm
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

After `ptxas -arch=sm_90`. This assembles the PTX assembly into the final binary format that can be executed on the GPU.

```asm
square_kernel:
 LDC R1, c[0x0][0x28] 
 S2R R3, SR_TID.X 
 LDC.64 R6, c[0x0][0x210] 
 ULDC.64 UR4, c[0x0][0x218] 
 ULDC.64 UR6, c[0x0][0x228] 
 S2R R0, SR_CTAID.X 
 IADD3 R2, P1, R3, UR4, RZ 
 IADD3 R5, P0, R0.reuse, R6, RZ 
 LEA.HI.X.SX32 R3, R3, UR5, 0x1, P1 
 ULDC.64 UR4, c[0x0][0x220] 
 LEA.HI.X.SX32 R0, R0, R7, 0x1, P0 
 IMAD.WIDE.U32 R2, R5, 0xa, R2 
 IMAD R7, R0, 0xa, RZ 
 IMAD.SHL.U32 R4, R2, 0x4, RZ 
 IMAD.IADD R3, R3, 0x1, R7 
 SHF.L.U64.HI R0, R2, 0x2, R3 
 IADD3 R2, P0, R4, UR4, RZ 
 IADD3.X R3, R0, UR5, RZ, P0, !PT 
 ULDC.64 UR4, c[0x0][0x208] 
 LDG.E R2, desc[UR4][R2.64] 
 IADD3 R4, P0, R4, UR6, RZ 
 IADD3.X R5, R0, UR7, RZ, P0, !PT 
 FMUL R7, R2, R2 
 STG.E desc[UR4][R4.64], R7 
 EXIT 
.L_x_0:
 BRA `(.L_x_0)
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
 NOP
.L_x_1:
```