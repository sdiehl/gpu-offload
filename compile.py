import subprocess

from mlir.ir import Context, Module
from mlir.passmanager import PassManager

module_str = """
module {
  func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %x0 : tensor<10x10xf32>
  }
}
"""


def gpu_frontend(ctx, module):
    pm = PassManager()
    pm.enable_ir_printing(print_after_change=True)
    pm.add("canonicalize")
    pm.add(
        "one-shot-bufferize{ bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map }"
    )
    pm.add("canonicalize")
    pm.add("convert-linalg-to-affine-loops")
    pm.add("func.func(affine-loop-invariant-code-motion)")
    pm.add("func.func(convert-affine-for-to-gpu)")
    pm.add("gpu-kernel-outlining")
    pm.add("lower-affine")
    pm.add("gpu-decompose-memrefs")
    pm.add("expand-strided-metadata")
    pm.add("normalize-memrefs")
    pm.add(
        "gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv })"
    )
    pm.add("nvvm-attach-target{chip=sm_90 features=+ptx80 O=3}")
    pm.add("convert-nvvm-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.add("gpu-to-llvm { use-bare-pointers-for-host use-bare-pointers-for-kernels }")
    # pm.add("gpu-module-to-binary")
    pm.run(module.operation)

    # Extract just the GPU module
    gpu_module = (
        module.operation.regions[0]
        .blocks[0]
        .operations[1]
        .regions[0]
        .blocks[0]
        .operations[0]
    )

    print("GPU Module:")
    gpu_module = Module.parse(str(gpu_module))
    print(str(gpu_module))

    # Print out to 'module.nvvmir'
    with open("module.nvvmir", "w") as f:
        f.write(str(gpu_module))

    return module, gpu_module


def generate_ptx(module_str):
    """Generate PTX from MLIR module string"""
    # First convert MLIR to LLVM IR

    # Write to output.mlir
    with open("output.mlir", "w") as f:
        f.write(module_str)

    llvm_ir_result = subprocess.run(
        ["mlir-translate", "--mlir-to-llvmir", "-"],
        input=module_str,
        capture_output=True,
        text=True,
    )

    if llvm_ir_result.returncode != 0:
        print("Error generating LLVM IR:")
        print(llvm_ir_result.stderr)
        return None

    llvm_ir = llvm_ir_result.stdout
    print("\nGenerated LLVM IR:")
    print(llvm_ir)

    # Then convert LLVM IR to PTX
    ptx_result = subprocess.run(
        ["llc", "-march=nvptx64", "-mcpu=sm_90", "-"],
        input=llvm_ir,
        capture_output=True,
        text=True,
    )

    if ptx_result.returncode != 0:
        print("Error generating PTX:")
        print(ptx_result.stderr)
        return None

    return ptx_result.stdout


def main():
    with Context() as ctx:
        module = Module.parse(module_str)
        print("Original Module:")
        print(str(module))

        module, gpu_module = gpu_frontend(ctx, module)
        print("\nTransformed Module:")
        print(str(module))

        # Generate PTX
        ptx = generate_ptx(str(gpu_module))
        if ptx:
            print("\nGenerated PTX:")
            print(ptx)
        else:
            print("\nFailed to generate PTX")

        # Write PTX to file
        with open("output.ptx", "w") as f:
            f.write(ptx)


if __name__ == "__main__":
    main()
