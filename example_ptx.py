import math
import numpy as np
from run import setup_cuda, run_ptx_kernel, cleanup_cuda, CudaError

# Example PTX code for vector addition
vector_add_ptx = """
.version 7.0 // Adjust version/target as needed
.target sm_70
.address_size 64

.visible .entry add_vectors(
    .param .u64 a_ptr, // .ptr .align 8 .const hints optional here
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n_elements
)
{
    // Simplified PTX - assumes float32
    .reg .u32 %tid, %n;
    .reg .u64 %a_addr, %b_addr, %c_addr, %offset;
    .reg .f32 %a_val, %b_val, %c_val;

    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %ctaid.x, %ntid.x, %tid; // Global thread ID

    ld.param.u32 %n, [n_elements];
    setp.ge.u32 %p, %tid, %n; // Bounds check
    @%p bra DONE;

    ld.param.u64 %a_addr, [a_ptr];
    ld.param.u64 %b_addr, [b_ptr];
    ld.param.u64 %c_addr, [c_ptr];

    mul.wide.u32 %offset, %tid, 4; // Offset for float32
    add.u64 %a_addr, %a_addr, %offset;
    add.u64 %b_addr, %b_addr, %offset;
    add.u64 %c_addr, %c_addr, %offset;

    ld.global.f32 %a_val, [%a_addr];
    ld.global.f32 %b_val, [%b_addr];
    add.f32 %c_val, %a_val, %b_val;
    st.global.f32 [%c_addr], %c_val;

DONE:
    ret;
}
"""

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
