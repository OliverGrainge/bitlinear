#!/bin/bash

# Force a clean rebuild of the C++ extension to avoid stale object files
# (e.g., old ukernel.o without i8dot_1x4) causing runtime symbol errors.
rm -rf build

python setup.py build_ext --inplace
pytest test.py 
python test_perf.py
