#!/bin/bash
set -exuo pipefail

# 检查当前用户是否为 root
if [ "$(id -u)" -ne 0 ]; then
    SUDO_CMD="sudo"
fi

$SUDO_CMD apt update || true
$SUDO_CMD apt install libcapstone-dev ninja-build || true

project_dir=$(git rev-parse --show-toplevel)
cd ${project_dir}

python setup.py bdist_wheel

ls -thl dist