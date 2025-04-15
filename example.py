import numpy as np
import math

from compile import compile_mlir_to_ptx
from run import setup_cuda, run_ptx_kernel, cleanup_cuda


# Example MLIR module for a matrix squaring operation
SQUARE_MLIR = """
module {
  func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %x0 : tensor<10x10xf32>
  }
}
"""


def run_square_example():
    """Demonstrates the full pipeline: MLIR to PTX compilation and GPU execution."""
    # Create input data
    size = 10
    input_data = np.random.randn(size, size).astype(np.float32)
    output_data = np.zeros((size, size), dtype=np.float32)

    # Expected result for verification
    expected_result = input_data * input_data

    # Step 1: Compile MLIR to PTX
    print("Compiling MLIR to PTX...")
    ptx_code = compile_mlir_to_ptx(SQUARE_MLIR)
    if not ptx_code:
        print("PTX compilation failed.")
        return

    context = None
    try:
        # Step 2: Setup CUDA
        print("Setting up CUDA...")
        context = setup_cuda()

        # Step 3: Run the kernel
        print("Running PTX kernel...")

        # Define the kernel configuration
        kernel_name = "square_kernel"  # This should match the name in the PTX
        arg_types = ["ptr:in", "ptr:out"]

        # Calculate grid/block dimensions
        threads_per_block = 16
        blocks_needed = math.ceil(size * size / threads_per_block)

        # Run the kernel
        run_ptx_kernel(
            ptx_code,
            kernel_name,
            arg_types,
            input_data,
            output_data,
            grid_dim=(blocks_needed,),
            block_dim=(threads_per_block,),
        )

        # Step 4: Verify results
        print("Verifying results...")
        np.testing.assert_allclose(output_data, expected_result, rtol=1e-5)
        print("Results verified successfully!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Step 5: Cleanup
        if context:
            cleanup_cuda(context)


if __name__ == "__main__":
    run_square_example()
