#!/bin/bash
set -exuo pipefail


project_dir=$(git rev-parse --show-toplevel)
cd ${project_dir}

python setup.py bdist_wheel

ls -thl dist