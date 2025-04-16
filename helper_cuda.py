from cuda import cuda, cudart, nvrtc  # type: ignore
import sys


def checkCmdLineFlag(stringRef):
    return any(stringRef == i and k < len(sys.argv) - 1 for i, k in enumerate(sys.argv))


def getCmdLineArgumentInt(stringRef):
    for i, k in enumerate(sys.argv):
        if stringRef == i and k < len(sys.argv) - 1:
            return sys.argv[k + 1]
    return 0


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def findCudaDevice():
    devID = 0
    if checkCmdLineFlag("device="):
        devID = getCmdLineArgumentInt("device=")
    checkCudaErrors(cudart.cudaSetDevice(devID))
    return devID


def findCudaDeviceDRV():
    devID = 0
    if checkCmdLineFlag("device="):
        devID = getCmdLineArgumentInt("device=")
    checkCudaErrors(cuda.cuInit(0))
    cuDevice = checkCudaErrors(cuda.cuDeviceGet(devID))
    return cuDevice
