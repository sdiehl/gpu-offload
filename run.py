import cuda.cuda as cu  # type: ignore
import numpy as np
import os
import tempfile
import subprocess
import ctypes
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


def cleanup_cuda(context):
    """Destroy the CUDA context."""
    if context:
        print("Destroying CUDA context...")
        checkCudaErrors(cu.cuCtxDestroy(context))
        print("CUDA context destroyed.")


def verify_ptx(ptx_code: str, chip_type: str = "sm_75") -> bool:
    """Verify PTX code using the NVIDIA PTX assembler (ptxas)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(ptx_code.encode("utf-8"))

        cmd = ["ptxas", f"-arch={chip_type}", temp_file_path]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )

        if result.returncode == 0:
            print(f"PTX verification successful ({chip_type}).")
            return True
        else:
            print(f"PTX verification failed ({chip_type}):")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print(
            "Error: ptxas command not found. Ensure CUDA toolkit is installed and in PATH."
        )
        return False
    except Exception as e:
        print(f"Error during PTX verification: {str(e)}")
        return False
    finally:
        os.unlink(temp_file_path)


class CudaArray:
    """Class to manage GPU memory and transfers with automatic cleanup."""

    def __init__(self, host_array=None, shape=None, dtype=np.float32):
        """Initialize a CUDA array either from host data or empty with given shape."""
        self.device_ptr = None
        self.nbytes = 0
        self.dtype = dtype
        self.shape = None

        # Create from host array
        if host_array is not None:
            if not isinstance(host_array, np.ndarray):
                host_array = np.array(host_array, dtype=dtype)
            self.host_array = host_array
            self.shape = host_array.shape
            self.nbytes = host_array.nbytes
            self.dtype = host_array.dtype
            self.device_ptr = checkCudaErrors(cu.cuMemAlloc(self.nbytes))
            self.copy_host_to_device()

        # Create empty with shape
        elif shape is not None:
            self.host_array = np.zeros(shape, dtype=dtype)
            self.shape = shape
            self.nbytes = self.host_array.nbytes
            self.device_ptr = checkCudaErrors(cu.cuMemAlloc(self.nbytes))

        else:
            raise ValueError("Either host_array or shape must be provided")

    def copy_host_to_device(self):
        """Copy data from host to device."""
        if not self.host_array.flags.c_contiguous:
            self.host_array = np.ascontiguousarray(self.host_array)
        checkCudaErrors(
            cu.cuMemcpyHtoD(self.device_ptr, self.host_array.ctypes.data, self.nbytes)
        )

    def copy_device_to_host(self):
        """Copy data from device to host."""
        checkCudaErrors(
            cu.cuMemcpyDtoH(self.host_array.ctypes.data, self.device_ptr, self.nbytes)
        )

    def free(self):
        """Free GPU memory."""
        if self.device_ptr:
            checkCudaErrors(cu.cuMemFree(self.device_ptr))
            self.device_ptr = None

    def __del__(self):
        """Automatically free GPU memory when object is deleted."""
        self.free()


class CudaContext:
    """Context manager for CUDA operations to ensure proper cleanup."""

    def __init__(self, device_id=0):
        self.context = None
        self.device_id = device_id
        self.arrays = []

    def __enter__(self):
        self.context = setup_cuda(self.device_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up any arrays that were created
        for array in self.arrays:
            array.free()

        # Destroy the CUDA context
        if self.context:
            cleanup_cuda(self.context)

    def array(self, host_array=None, shape=None, dtype=np.float32):
        """Create a CudaArray and register it for automatic cleanup."""
        array = CudaArray(host_array, shape, dtype)
        self.arrays.append(array)
        return array

    def run_kernel(
        self,
        ptx_code,
        kernel_name,
        args,
        n=None,
        grid_dims=None,
        block_dims=(128, 1, 1),
    ):
        """Run a PTX kernel with automatic dimension calculation if needed."""
        # Prepare arguments
        arg_values = []
        arg_types = []

        for arg in args:
            if isinstance(arg, CudaArray):
                arg_values.append(arg.device_ptr)
                arg_types.append(None)  # None for void pointers
            elif isinstance(arg, int):
                arg_values.append(arg)
                arg_types.append(ctypes.c_int)
            elif isinstance(arg, float):
                arg_values.append(arg)
                arg_types.append(ctypes.c_float)
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Prepare grid and block dimensions
        if grid_dims is not None:
            # Use explicit grid dimensions
            grid = list(grid_dims) + [1] * (3 - len(grid_dims))
            block = list(block_dims) + [1] * (3 - len(block_dims))
        elif n is not None:
            # Calculate grid dimensions from n
            threads_per_block = (
                block_dims[0] if isinstance(block_dims, tuple) else block_dims
            )
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            grid = [blocks_per_grid, 1, 1]
            block = [threads_per_block, 1, 1]
        else:
            raise ValueError("Either n or grid_dims must be provided")

        # Load the PTX module
        module = checkCudaErrors(cu.cuModuleLoadData(ptx_code.encode("utf-8")))

        # Get kernel function
        kernel_func = checkCudaErrors(
            cu.cuModuleGetFunction(module, kernel_name.encode("utf-8"))
        )

        # Create kernel args tuple
        kernel_args = (tuple(arg_values), tuple(arg_types))

        # Launch the kernel
        checkCudaErrors(
            cu.cuLaunchKernel(
                kernel_func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,  # shared memory bytes
                0,  # stream
                kernel_args,  # kernel args
                0,  # extra
            )
        )

        # Synchronize to ensure kernel completion
        checkCudaErrors(cu.cuCtxSynchronize())

        # Unload module when done
        checkCudaErrors(cu.cuModuleUnload(module))
