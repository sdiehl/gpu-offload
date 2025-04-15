import cuda.cuda as cu
import numpy as np
import inspect  # For error messages


# --- CUDA Error Checking (Minimal) ---
class CudaError(RuntimeError):
    """Custom exception for CUDA errors."""

    def __init__(self, message, cu_result):
        super().__init__(f"{message} (CUDA Error: {cu_result})")
        self.cu_result = cu_result


def _check_cuda_error(err_tuple, func_name="<unknown>"):
    """Checks CUDA Driver API error codes."""
    if not isinstance(err_tuple, tuple):
        # Handle cases where only the error code is returned directly
        if isinstance(err_tuple, cu.CUresult) and err_tuple != cu.CUresult.CUDA_SUCCESS:
            err = err_tuple
        else:
            return  # No error or not a CUresult

    elif err_tuple[0] != cu.CUresult.CUDA_SUCCESS:
        err = err_tuple[0]  # Error code is usually the first element
    else:
        return  # Success

    # Error occurred, raise exception
    try:
        err_name = cu.cuGetErrorName(err)[1].decode("utf-8")
    except Exception:
        err_name = "Unknown Error Name"
    try:
        err_string = cu.cuGetErrorString(err)[1].decode("utf-8")
    except Exception:
        err_string = "Unknown Error Description"

    caller = inspect.currentframe().f_back
    caller_info = f"in '{caller.f_code.co_name}' line {caller.f_lineno}"
    raise CudaError(
        f"CUDA call {func_name}(...) failed {caller_info}. {err_name}: {err_string}",
        err,
    )


# --- Type Mapping (Simplified) ---
# Maps simple type names to numpy types for kernel arguments. 'ptr' is always uint64.
# We'll enforce that 'ptr' corresponds to float32 np.ndarray elsewhere.
_SCALAR_NP_TYPES = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
    "ptr": np.uint64,  # PTX kernels take pointers as 64-bit unsigned integers
}

# --- Core Functions ---


def setup_cuda(device_id: int = 0):
    """Initializes CUDA and creates a context."""
    print("Initializing CUDA...")
    _check_cuda_error(cu.cuInit(0), "cuInit")
    err, device = cu.cuDeviceGet(device_id)
    _check_cuda_error((err,), "cuDeviceGet")  # Wrap single return in tuple for checker
    err, context = cu.cuCtxCreate(0, device)
    _check_cuda_error((err,), "cuCtxCreate")
    print(f"CUDA context created on device {device_id}.")
    # Context is now active on the calling thread
    return context  # Return context handle needed for cleanup


def run_ptx_kernel(
    ptx_code: str,
    kernel_name: str,
    arg_types: list[str],
    *args,  # Kernel arguments must come before named args below
    grid_dim: tuple[int, ...],
    block_dim: tuple[int, ...],
):
    """Runs a PTX kernel on the GPU.

    Args:
        ptx_code: String containing PTX code
        kernel_name: Name of the kernel function to call
        arg_types: List of type strings for kernel arguments (e.g., ["ptr:in", "ptr:out", "int32"])
        *args: Kernel arguments (numpy arrays or scalar values)
        grid_dim: Grid dimensions as tuple (e.g., (8, 1, 1))
        block_dim: Block dimensions as tuple (e.g., (128, 1, 1))

    Returns:
        List of output arrays that were modified by the kernel
    """
    module = None
    gpu_allocations = []  # Stores (gpu_ptr, host_array_for_output) for cleanup/copyback
    output_arrays = []  # References to host arrays that need updating

    try:
        # 1. Load PTX Module
        # print("Loading PTX module...")
        err, module = cu.cuModuleLoadData(ptx_code.encode("utf-8"))
        _check_cuda_error((err,), "cuModuleLoadData")

        # 2. Get Kernel Function
        err, kernel_func = cu.cuModuleGetFunction(module, kernel_name.encode("utf-8"))
        _check_cuda_error((err,), "cuModuleGetFunction")

        # 3. Prepare Arguments (Allocate GPU memory, Copy H->D)
        kernel_args_list = []
        if len(args) != len(arg_types):
            raise ValueError(
                f"Expected {len(arg_types)} kernel arguments (*args), got {len(args)}"
            )

        for i, (type_str, host_arg) in enumerate(zip(arg_types, args)):
            parts = type_str.lower().split(":")
            base_type = parts[0]
            modifier = parts[1] if len(parts) > 1 else "in"

            if base_type == "ptr":
                if not isinstance(host_arg, np.ndarray) or host_arg.dtype != np.float32:
                    raise TypeError(
                        f"Arg {i}: 'ptr' type expects a float32 NumPy array, got {type(host_arg)} with dtype {getattr(host_arg, 'dtype', None)}"
                    )
                if not host_arg.flags["C_CONTIGUOUS"]:
                    host_arg = np.ascontiguousarray(host_arg)  # Ensure contiguous

                # Allocate GPU memory
                err, gpu_ptr = cu.cuMemAlloc(host_arg.nbytes)
                _check_cuda_error((err,), "cuMemAlloc")

                needs_copy_to_gpu = modifier in ["in", "inout"]
                needs_copy_back = modifier in ["out", "inout"]

                # Copy Host -> Device if needed
                if needs_copy_to_gpu:
                    (err,) = cu.cuMemcpyHtoD(
                        gpu_ptr, host_arg.ctypes.data, host_arg.nbytes
                    )
                    _check_cuda_error((err,), "cuMemcpyHtoD")

                # Store pointer for kernel args and info for cleanup/copyback
                kernel_args_list.append(_SCALAR_NP_TYPES["ptr"](gpu_ptr))
                gpu_allocations.append((gpu_ptr, host_arg if needs_copy_back else None))
                if needs_copy_back:
                    output_arrays.append(host_arg)

            elif base_type in _SCALAR_NP_TYPES:  # Scalar argument
                np_type = _SCALAR_NP_TYPES[base_type]
                try:
                    scalar_val = np_type(host_arg)
                except (TypeError, ValueError):
                    raise TypeError(
                        f"Arg {i}: Cannot convert '{host_arg}' to scalar type '{base_type}'"
                    )
                kernel_args_list.append(scalar_val)
            else:
                raise ValueError(f"Arg {i}: Unsupported base type '{base_type}'")

        # 4. Launch Kernel
        # print(f"Launching kernel '{kernel_name}'...")
        grid = grid_dim + (1,) * (3 - len(grid_dim))
        block = block_dim + (1,) * (3 - len(block_dim))

        (err,) = cu.cuLaunchKernel(
            kernel_func,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            0,
            0,  # Shared mem bytes, stream handle (0=default)
            args=kernel_args_list,
        )
        _check_cuda_error((err,), "cuLaunchKernel")

        # 5. Synchronize (Wait for kernel completion)
        (err,) = cu.cuCtxSynchronize()
        _check_cuda_error((err,), "cuCtxSynchronize")
        # print("Kernel finished.")

        # 6. Copy Results Device -> Host
        for gpu_ptr, host_array_out in gpu_allocations:
            if host_array_out is not None:  # If marked for copy-back
                # print("Copying results D->H...")
                (err,) = cu.cuMemcpyDtoH(
                    host_array_out.ctypes.data, gpu_ptr, host_array_out.nbytes
                )
                _check_cuda_error((err,), "cuMemcpyDtoH")

        return output_arrays  # Return the modified host arrays

    finally:
        # 7. Cleanup Resources for this run (Memory, Module)
        # print("Cleaning up kernel resources...")
        for gpu_ptr, _ in gpu_allocations:
            if gpu_ptr:
                try:
                    (err,) = cu.cuMemFree(gpu_ptr)
                    # Don't check error strictly here during cleanup to ensure all attempts are made
                    if err != cu.CUresult.CUDA_SUCCESS:
                        print(
                            f"Warning: cuMemFree failed for pointer {gpu_ptr} with error {err}"
                        )
                except Exception as e:  # Catch potential exceptions during cleanup
                    print(
                        f"Warning: Exception during cuMemFree for pointer {gpu_ptr}: {e}"
                    )
        if module:
            try:
                (err,) = cu.cuModuleUnload(module)
                if err != cu.CUresult.CUDA_SUCCESS:
                    print(f"Warning: cuModuleUnload failed with error {err}")
            except Exception as e:
                print(f"Warning: Exception during cuModuleUnload: {e}")


def cleanup_cuda(context):
    """Destroys the CUDA context."""
    if context:
        print("Destroying CUDA context...")
        try:
            (err,) = cu.cuCtxDestroy(context)
            # Don't check error strictly during cleanup
            if err != cu.CUresult.CUDA_SUCCESS:
                print(f"Warning: cuCtxDestroy failed with error {err}")
            else:
                print("CUDA context destroyed.")
        except Exception as e:
            print(f"Warning: Unexpected error during context destruction: {e}")
