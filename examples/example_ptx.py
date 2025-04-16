import sys

sys.path.append(".")

import numpy as np
from run import CudaContext

# Example PTX code for vector addition
vector_add_ptx = open("examples/vecAdd.ptx").read()

# Prepare data
n = 10000

# Using context manager for automatic resource management
with CudaContext() as ctx:
    print("CUDA initialized. Running vector addition example...")

    # Prepare input arrays
    h_A = np.random.rand(n).astype(dtype=np.float32)
    h_B = np.random.rand(n).astype(dtype=np.float32)

    # Create CudaArray objects using the context - these handle memory allocation and transfers
    d_A = ctx.array(h_A)
    d_B = ctx.array(h_B)
    d_C = ctx.array(shape=n, dtype=np.float32)  # Empty output array

    # Run kernel
    print("Running kernel...")
    ctx.run_kernel(
        vector_add_ptx, "add_vectors", [d_A, d_B, d_C, n], n=n, block_dims=(128, 1, 1)
    )

    print("Kernel execution complete.")

    # Copy results back to host
    d_C.copy_device_to_host()
    h_C = d_C.host_array

    # Memory will be automatically freed when context exits

    # --- Verify ---
    print("Verifying results...")
    correct = True
    for i in range(n):
        expected = h_A[i] + h_B[i]
        if abs(h_C[i] - expected) > 1e-7:
            correct = False
            print(f"Error at index {i}: {h_C[i]} != {expected}")
            break

    if correct:
        print("Results verified successfully!")
    else:
        print("Result verification failed!")
