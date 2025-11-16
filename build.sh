#!/bin/bash

python setup.py build_ext --inplace
pytest test.py 
python test_perf.py
