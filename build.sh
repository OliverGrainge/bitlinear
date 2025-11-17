#!/bin/bash

# Force a clean rebuild of the C++ extension to avoid stale object files
# (e.g., old ukernel.o without i8dot_1x4) causing runtime symbol errors.
rm -rf build dist *.egg-info

# Install build tools if not already installed
pip install build wheel --quiet

# Build wheel and source distribution for local testing
python -m build

# Install locally for testing (reinstall to pick up changes)
# Use wheel for faster local iteration
pip install dist/bitlinear-*.whl --force-reinstall

# Verify the extension was built
python -c "import _bitlinear; print('✓ Extension built successfully!')" || {
    echo "ERROR: Extension not found! Build may have failed."
    exit 1
}

# Run tests
pytest test.py 
python test_perf.py
