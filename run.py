import cuda.cuda as cu # type: ignore
import numpy as np
import os
import tempfile
import subprocess
from helper_cuda import checkCudaErrors


def setup_cuda(device_id=0):
    """Initialize CUDA and create a context."""
    print("Initializing CUDA...")
    # Initialize CUDA
    checkCudaErrors(cu.cuInit(0))
    
    # Get device
    device = checkCudaErrors(cu.cuDeviceGet(device_id))
    
    # Create context
    context = checkCudaErrors(cu.cuCtxCreate(0, device))
    
    print(f"CUDA context created on device {device_id}.")
    return context


def run_ptx_kernel(
    ptx_code: str,
    kernel_name: str,
    args_data: list,
    args_types: list,
    grid_dim: tuple,
    block_dim: tuple
):
    """Run a PTX kernel with simplified argument handling.
    
    Args:
        ptx_code: String containing PTX code
        kernel_name: Name of the kernel function to call
        args_data: List of kernel arguments (numpy arrays or scalars)
        args_types: List of ctypes types for kernel arguments
        grid_dim: Grid dimensions as tuple (e.g., (8, 1, 1))
        block_dim: Block dimensions as tuple (e.g., (128, 1, 1))
    """
    # Load the PTX module
    module = checkCudaErrors(cu.cuModuleLoadData(ptx_code.encode("utf-8")))
    
    # Get kernel function
    kernel_func = checkCudaErrors(cu.cuModuleGetFunction(module, kernel_name.encode("utf-8")))
    
    # Prepare kernel arguments
    kernel_args = (tuple(args_data), tuple(args_types))
    
    # Prepare grid and block dimensions
    grid = tuple(int(x) for x in grid_dim) + (1,) * (3 - len(grid_dim))
    block = tuple(int(x) for x in block_dim) + (1,) * (3 - len(block_dim))
    
    # Launch the kernel
    checkCudaErrors(cu.cuLaunchKernel(
        kernel_func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        0,                     # shared memory bytes
        0,                     # stream
        kernel_args,
        0                      # extra
    ))
    
    # Synchronize to ensure kernel completion
    checkCudaErrors(cu.cuCtxSynchronize())
    
    # Unload module when done
    checkCudaErrors(cu.cuModuleUnload(module))


def cleanup_cuda(context):
    """Destroy the CUDA context."""
    if context:
        print("Destroying CUDA context...")
        checkCudaErrors(cu.cuCtxDestroy(context))
        print("CUDA context destroyed.")


def verify_ptx(ptx_code: str, arch: int = 75) -> bool:
    """Verify PTX code using the NVIDIA PTX assembler (ptxas)."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(ptx_code.encode('utf-8'))
            
        cmd = ["ptxas", f"-arch=sm_{arch}", temp_file_path]
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(f"PTX verification successful (SM_{arch}).")
            return True
        else:
            print(f"PTX verification failed (SM_{arch}):")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("Error: ptxas command not found. Ensure CUDA toolkit is installed and in PATH.")
        return False
    except Exception as e:
        print(f"Error during PTX verification: {str(e)}")
        return False
    finally:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


# Helper function for allocating and managing device memory
def cuda_malloc_and_copy(host_array):
    """Allocate GPU memory and copy data from host to device."""
    if not isinstance(host_array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(host_array)}")
    
    # Ensure contiguous array
    if not host_array.flags["C_CONTIGUOUS"]:
        host_array = np.ascontiguousarray(host_array)
    
    # Allocate GPU memory
    device_ptr = checkCudaErrors(cu.cuMemAlloc(host_array.nbytes))
    
    # Copy data from host to device
    checkCudaErrors(cu.cuMemcpyHtoD(device_ptr, host_array.ctypes.data, host_array.nbytes))
    
    return device_ptr, host_array.nbytes


def cuda_memcpy_device_to_host(device_ptr, host_array):
    """Copy data from device to host."""
    checkCudaErrors(cu.cuMemcpyDtoH(host_array.ctypes.data, device_ptr, host_array.nbytes))


def cuda_free(device_ptr):
    """Free GPU memory."""
    checkCudaErrors(cu.cuMemFree(device_ptr))
