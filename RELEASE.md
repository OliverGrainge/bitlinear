# Release Instructions

This document describes how to create releases on GitHub.

## Creating a GitHub Release

1. **Update version** in `setup.py` and `pyproject.toml`:
   ```python
   version="0.1.0"  # Update to new version
   ```

2. **Update CHANGELOG.md** (if you have one) with release notes

3. **Commit and tag the release**:
   ```bash
   git add setup.py pyproject.toml
   git commit -m "Bump version to 0.1.0"
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin main
   git push origin v0.1.0
   ```

4. **Create a GitHub Release**:
   - Go to https://github.com/oliver/bitlinear/releases/new
   - Select the tag you just pushed
   - Add release notes describing changes
   - Publish the release

## Installation Instructions for Users

Users can install from GitHub using:

```bash
git clone https://github.com/oliver/bitlinear.git
cd bitlinear
pip install .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/oliver/bitlinear.git
```

For a specific version/tag:

```bash
pip install git+https://github.com/oliver/bitlinear.git@v0.1.0
```

## Build Requirements

Users need to have:
- Python >= 3.7
- PyTorch >= 1.13.0 (installed before building)
- C++ compiler (gcc/clang on Linux/Mac, MSVC on Windows)
- CUDA toolkit (optional, for GPU support)
- OpenMP (usually included with compiler)

The extension will be compiled during installation, which may take a few minutes.
