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

if not ptx_code:
    raise RuntimeError("PTX compilation failed.")

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
    print("Verifying results...")
    np.testing.assert_allclose(result, expected_output, rtol=1e-5)
    print("Success! Results verified.")
