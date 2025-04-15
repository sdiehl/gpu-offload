# GPU Offload Tutorial

This minimal project demonstrates how to compile MLIR code to PTX and execute it on NVIDIA GPUs using Python.

## Project Structure

- `compile.py`: Functions for compiling MLIR to PTX
- `run.py`: Functions for running PTX kernels on CUDA GPUs
- `example.py`: Example showing the complete pipeline

## Requirements

- Python 3.12+
- MLIR Python bindings
- CUDA Python bindings
- NumPy

## Installation

```bash
poetry install
```

## Usage

The example shows a complete pipeline for compiling and running a simple square operation:

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
```
