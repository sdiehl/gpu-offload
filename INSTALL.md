# Installation

The systems dependencies for MLIR and CUDA are somewhat involved.

1. **You need a Linux system, MacOS is not supported.**
2. **You need a NVIDIA GPU attached to your system.**

The easiest way to get started is using the NVIDIA CUDA Docker image:

```bash
docker pull nvidia/cuda:12.1.0-devel-ubuntu22.04
```

1. Start a container with the image, pass the `--gpus` flag to enable host GPU access:

```bash
docker run -it --gpus all nvidia/cuda:12.1.0-devel-ubuntu22.04 bash
```

2. Inside the container, install system dependencies:

```bash
apt-get update
apt-get install -y wget gnupg software-properties-common python3.10 python3.10-dev python3.10-venv python3-pip git
```

3. Install LLVM and MLIR tools:

```bash
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
add-apt-repository -y "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main"
apt-get update
apt-get install -y llvm-20 llvm-20-dev llvm-20-tools mlir-20-tools
ln -sf /usr/bin/llc-20 /usr/bin/llc
ln -sf /usr/bin/mlir-translate-20 /usr/bin/mlir-translate
```

4. Install Poetry:

```bash
python3.10 -m pip install pip --upgrade
python3.10 -m pip install poetry
```

5. Clone and set up the project:

```bash
git clone https://github.com/sdiehl/gpu-offload.git
cd gpu-offload
poetry env use python3.10
poetry install --with gpu
```
