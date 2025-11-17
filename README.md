# BitLinear

Efficient BitLinear implementation with CPU and CUDA support for PyTorch.

## Description

BitLinear is a binary neural network linear layer implementation that quantizes weights to ternary values `{-1, 0, 1}` and activations to int8. This package provides optimized CPU and CUDA kernels for efficient inference.

## Features

- **CPU and CUDA support**: Optimized kernels for both CPU (with OpenMP) and CUDA
- **Training and deployment modes**: Supports both training with gradient flow and efficient inference
- **Flexible quantization**: Configurable quantization types for different use cases
- **PyTorch integration**: Seamless integration with PyTorch's `nn.Module` API

## Installation

### Prerequisites

Before installing, ensure you have:

- **Python** >= 3.7
- **PyTorch** >= 1.13.0 (install from [pytorch.org](https://pytorch.org/get-started/locally/))
- **C++ compiler**:
  - Linux: `gcc` or `clang` (usually pre-installed)
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Windows: Visual Studio Build Tools or MSVC
- **CUDA toolkit** (optional, for GPU support)
- **OpenMP** (for CPU parallelization, usually included with compiler)

### Install from GitHub

**Option 1: Clone and install**
```bash
git clone https://github.com/oliver/bitlinear.git
cd bitlinear
pip install .
```

**Option 2: Install directly from GitHub** (no clone needed)
```bash
pip install git+https://github.com/oliver/bitlinear.git
```

For a specific version/tag:
```bash
pip install git+https://github.com/oliver/bitlinear.git@v0.1.0
```

The C++/CUDA extension will be compiled during installation. This may take a few minutes.

### CPU-only Build

To skip CUDA compilation (faster build, CPU-only):

```bash
BITLINEAR_FORCE_CPU=1 pip install .
```

### Verify Installation

After installation, verify the extension was built successfully:

```bash
python -c "import _bitlinear; print('Installation successful!')"
```

## Usage

### Basic Usage

```python
import torch
from bitlinear import BitLinear

# Create a BitLinear layer
layer = BitLinear(in_features=512, out_features=256, bias=True)

# Forward pass (training mode)
x = torch.randn(32, 512)
output = layer(x)
```

### Deployment Mode

For efficient inference, deploy the layer to use optimized kernels:

```python
# Deploy for inference (quantizes and packs weights)
layer.deploy()

# Run inference
with torch.no_grad():
    output = layer(x)
```

### Converting from Standard Linear Layer

```python
from torch import nn
from bitlinear import BitLinear

# Create a standard linear layer
linear = nn.Linear(512, 256)

# Convert to BitLinear
bitlinear = BitLinear.from_linear(linear, quant_type="ai8pc_wpt")
```

## Development

### Building Locally

For development, you can build and test locally:

```bash
# Clean build and test
./build.sh

# Or build manually
pip install -e .
```

### Testing

```bash
pytest test.py
python test_perf.py
```

### Troubleshooting Installation

If installation fails, check:

1. **PyTorch is installed**: The C++ source files require PyTorch headers (`<torch/extension.h>`)
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. **C++ compiler is available**:
   - Linux: `gcc --version` or `clang --version`
   - macOS: `clang --version`
   - Windows: Check Visual Studio installation

3. **CUDA (if using GPU)**: Ensure CUDA toolkit matches your PyTorch CUDA version
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

4. **Build errors**: Check the full error output. Common issues:
   - Missing compiler
   - Missing OpenMP library
   - CUDA version mismatch
   - PyTorch version incompatibility

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

