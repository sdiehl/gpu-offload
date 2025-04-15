from compile import compile_mlir_to_ptx

# Example MLIR module for a matrix squaring operation
SQUARE_MLIR = """
module {
  func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %x0 : tensor<10x10xf32>
  }
}
"""


def main():
    # Compile the MLIR module to PTX
    ptx_code = compile_mlir_to_ptx(SQUARE_MLIR)

    # Print the generated PTX code
    print(ptx_code)


if __name__ == "__main__":
    main()
