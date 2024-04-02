#!/bin/bash
set -exuo pipefail

# 定义 SUDO_CMD 变量的默认值
SUDO_CMD=""

# 检查当前用户是否为 root，并设置 SUDO_CMD 变量
if [ "$(id -u)" -ne 0 ]; then
    SUDO_CMD="sudo"
fi

# 使用 SUDO_CMD 执行 apt 命令
$SUDO_CMD apt update || true
$SUDO_CMD apt install libcapstone-dev ninja-build || true

project_dir=$(git rev-parse --show-toplevel)
cd ${project_dir}

python setup.py bdist_wheel

ls -thl dist