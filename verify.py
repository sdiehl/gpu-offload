import os
import subprocess
import tempfile


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
