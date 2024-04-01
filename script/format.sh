#!/bin/bash
set -exuo pipefail


project_dir=$(git rev-parse --show-toplevel)
cd ${project_dir}

find . -name "*.py" -exec black {} +
echo "All Python files formatted using Black."