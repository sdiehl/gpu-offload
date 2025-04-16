import numpy as np
import ctypes
from run import setup_cuda, run_ptx_kernel, cleanup_cuda
from helper_cuda import checkCudaErrors
import cuda.cuda as cu  # type: ignore

# Example PTX code for vector addition
vector_add_ptx = open("example.ptx").read()

if __name__ == "__main__":
    # Prepare data
    n = 10000
    cuda_context = None  # Ensure cleanup happens even if setup fails

    try:
        # --- Setup ---
        cuda_context = setup_cuda()

        # Allocate memory normally - we'll use Driver API only
        h_A = np.random.rand(n).astype(dtype=np.float32)
        h_B = np.random.rand(n).astype(dtype=np.float32)
        h_C = np.zeros(n, dtype=np.float32)
        nbytes = n * np.dtype(np.float32).itemsize

        # --- Define kernel dimensions ---
        threadsPerBlock = 128
        blocksPerGrid = (n + threadsPerBlock - 1) // threadsPerBlock

        # Allocate device memory using Driver API
        d_A = checkCudaErrors(cu.cuMemAlloc(nbytes))
        d_B = checkCudaErrors(cu.cuMemAlloc(nbytes))
        d_C = checkCudaErrors(cu.cuMemAlloc(nbytes))

        # Copy input data to device
        checkCudaErrors(cu.cuMemcpyHtoD(d_A, h_A, nbytes))
        checkCudaErrors(cu.cuMemcpyHtoD(d_B, h_B, nbytes))

        # --- Run kernel ---
        print("Running kernel...")

        # Note: None for void pointers, explicit type for scalar parameters
        kernelArgs = ((d_A, d_B, d_C, n), (None, None, None, ctypes.c_int))

        run_ptx_kernel(
            vector_add_ptx,
            "add_vectors",
            kernelArgs[0],  # Values
            kernelArgs[1],  # Types
            (threadsPerBlock,),
            (blocksPerGrid,),
        )
        print("Kernel execution complete.")

        # Copy results back to host
        checkCudaErrors(cu.cuMemcpyDtoH(h_C, d_C, nbytes))

        # Free device memory
        checkCudaErrors(cu.cuMemFree(d_A))
        checkCudaErrors(cu.cuMemFree(d_B))
        checkCudaErrors(cu.cuMemFree(d_C))

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

    except RuntimeError as e:
        print(f"\n*** CUDA error: {e} ***")
    except Exception as e:
        print(f"\n*** An unexpected error occurred: {e} ***")
        import traceback

        traceback.print_exc()  # Print stack trace for unexpected errors
    finally:
        # --- Cleanup ---
        if cuda_context:
            cleanup_cuda(cuda_context)

    print("Example finished.")
