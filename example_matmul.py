import numpy as np
from run import CudaContext

# Define PTX code for matrix multiplication
matrix_multiply_ptx = open("matmul.ptx").read()

# Matrix dimensions
N = 64  # Rows of A
K = 128  # Columns of A, Rows of B
M = 64  # Columns of B

# Using context manager for automatic resource management
with CudaContext() as ctx:
    print("CUDA initialized. Running matrix multiplication example...")

    # Create random matrices
    A = np.random.rand(N, K).astype(np.float32)
    B = np.random.rand(K, M).astype(np.float32)

    # Calculate expected result on CPU for verification
    expected_C = np.matmul(A, B)

    # Create CUDA arrays using the context
    d_A = ctx.array(A)
    d_B = ctx.array(B)
    d_C = ctx.array(shape=(N, M), dtype=np.float32)

    # Define block and grid dimensions
    threads_per_block = 128
    blocks_x = (M + threads_per_block - 1) // threads_per_block
    blocks_y = N

    # Run the kernel using the context
    ctx.run_kernel(
        matrix_multiply_ptx,
        "matrixMul",
        [d_A, d_B, d_C, N, M, K],
        grid_dims=(blocks_x, blocks_y),
        block_dims=(threads_per_block, 1, 1),
    )

    # Copy results back to host
    d_C.copy_device_to_host()
    result_C = d_C.host_array

    # Verify results
    max_error = np.max(np.abs(result_C - expected_C))
    print(f"Maximum error: {max_error}")

    if max_error < 1e-5:
        print("Matrix multiplication successful!")
    else:
        print("Matrix multiplication failed: error too large")
