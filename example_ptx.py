import math
import numpy as np
from run import setup_cuda, run_ptx_kernel, cleanup_cuda, CudaError

# Example PTX code for vector addition
vector_add_ptx = open("example.ptx").read()

if __name__ == "__main__":
    # Prepare data
    n = 10000
    a_host = np.random.randn(n).astype(np.float32)
    b_host = np.random.randn(n).astype(np.float32)
    c_host = np.empty_like(a_host)  # Output array

    print(f"Input array size: {n}")
    cuda_context = None  # Ensure cleanup happens even if setup fails

    try:
        # --- Setup ---
        cuda_context = setup_cuda()

        # --- Define kernel signature and dimensions ---
        arg_signature = ["ptr:in", "ptr:in", "ptr:out", "int32"]
        threads_per_block = 128
        blocks_per_grid = math.ceil(n / threads_per_block)

        # --- Run ---
        print("Running kernel...")
        # --- CORRECTED CALL ---
        # Positional args first (ptx, name, types, *kernel_args)
        # Keyword args last (grid_dim, block_dim)
        output_arrays = run_ptx_kernel(
            vector_add_ptx,
            "add_vectors",
            arg_signature,
            # --- Kernel arguments collected by *args ---
            a_host,  # ptr:in (float32 array)
            b_host,  # ptr:in (float32 array)
            c_host,  # ptr:out (float32 array)
            np.int32(n),  # int32 scalar
            # --- Keyword arguments ---
            grid_dim=(blocks_per_grid,),
            block_dim=(threads_per_block,),
        )
        print("Kernel execution complete.")

        # output_arrays contains references to the modified host arrays (c_host here)
        assert output_arrays[0] is c_host

        # --- Verify ---
        print("Verifying results...")
        expected_c = a_host + b_host
        np.testing.assert_allclose(c_host, expected_c, rtol=1e-6)
        print("Results verified successfully!")

    except CudaError as e:
        print(f"\n*** A CUDA error occurred: {e} ***")
    except Exception as e:
        print(f"\n*** An unexpected error occurred: {e} ***")
        import traceback

        traceback.print_exc()  # Print stack trace for unexpected errors
    finally:
        # --- Cleanup ---
        if cuda_context:
            cleanup_cuda(cuda_context)

    print("Example finished.")
