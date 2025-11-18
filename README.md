# BitLinear

Efficient Memory and Latency optimized BitLinear implementation with CPU and CUDA support for PyTorch.

## Latency Overview

![Latency Comparison](profile/plots/latency_comparison.png)

BitLinear's execution path is built around **2-bit weight packing**, fused quantized arithmetic, and hardware-specific vectorization. All kernels are aggressively tuned for low-latency inference using:

- **ARM NEON** and **NEON DotProd**
- **x86 AVX2**, **AVX-VNNI**, **AVX512-VNNI**
- **CUDA** bit-packed GPU kernels

These implementations reduce memory bandwidth pressure while maximizing arithmetic intensity, yielding the latency characteristics shown above.

---

## Description

BitLinear is a binary / ternary neural network linear layer implementation that quantizes weights to `{−1, 0, 1}` and activations to int8. This package provides optimized CPU and CUDA kernels for efficient inference.

## Features

- **Highly-optimized kernels**
  - 2-bit weight packing for minimal bandwidth
  - Hardware-specific SIMD kernels: NEON, AVX2, AVX-VNNI, AVX512-VNNI
  - CUDA kernels for GPU bit-packed matrix multiplication
- **CPU and CUDA support**: Optimized backends for both CPU (with OpenMP) and CUDA
- **Training and deployment modes**: Supports both training with gradient flow and efficient inference
- **Flexible quantization**: Multiple quantization types for different model families
- **PyTorch integration**: Seamless integration with PyTorch's `nn.Module` API

## Performance

### Latency Comparison

![Latency Comparison](profile/plots/latency_comparison.png)

### Throughput Comparison

![Throughput Comparison](profile/plots/throughput_comparison.png)

### Static Memory

<img src="profile/plots/static_memory.png" alt="Static Memory" width="400"/>

## Quick Start

```bash
git clone https://github.com/OliverGrainge/bitlinear.git
cd bitlinear
pip install -e .
python -c "import _bitlinear; print('✓ Kernels built successfully!')"
```

## Installation

### Prerequisites

- Python >= 3.7
- PyTorch >= 1.13.0
- C++ compiler (gcc/clang/MSVC)
- CUDA toolkit (optional, for GPU support)
- OpenMP (for CPU parallelization)

### Building the Kernels

Kernels are compiled during installation.

#### Editable Install (Recommended)

```bash
git clone https://github.com/OliverGrainge/bitlinear.git
cd bitlinear
pip install -e .
```

#### Regular Install

```bash
pip install .
```

During installation BitLinear will:
1. Detect CUDA availability
2. Build CPU kernels
3. Build CUDA kernels (if available)
4. Produce the `_bitlinear` extension module

#### Verify the Kernels Were Built

```bash
python -c "import _bitlinear; print('✓ Kernels built successfully!')"
```

## Troubleshooting Build Issues

1. Verify PyTorch installation  
2. Verify compiler availability  
3. Inspect build output  
4. Force clean rebuild  
5. Confirm `_bitlinear` extension exists  
6. Use verbose install mode  
7. On macOS, fix missing rpath if needed  

## Usage

### Basic Usage

```python
import torch
from bitlinear import BitLinear

layer = BitLinear(in_features=512, out_features=256, bias=True)
x = torch.randn(32, 512)
output = layer(x)
```

### Deployment Mode

```python
layer.deploy()
with torch.no_grad():
    output = layer(x)
```

### Convert from Standard Linear

```python
from torch import nn
from bitlinear import BitLinear

linear = nn.Linear(512, 256)
bitlinear = BitLinear.from_linear(linear, quant_type="ai8pc_wpt")
```

## Development

### Build Locally

```bash
./build.sh
pip install -e .
```

### Testing

```bash
pytest test.py
python test_perf.py
```

## License

MIT License

## Contributing

Contributions are welcome! Please submit a Pull Request.
