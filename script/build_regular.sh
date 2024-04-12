#!/bin/bash
set -exuo pipefail

# NOTE:
# This script is used to build the wheel package which just profiling XPU kernel.
# Not support GPU device profiling

project_dir=$(git rev-parse --show-toplevel)
cd ${project_dir}

#just compile and package the python code
export COMPIEL_OPTION=True
python setup.py bdist_wheel

ls -thl dist