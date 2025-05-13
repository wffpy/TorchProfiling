#!/bin/bash
set -euo pipefail  # 更安全（-x 可按需开启调试）

#================= [1] 设置 SUDO 命令 =================#
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO_CMD="sudo"
fi

#================= [2] 安装依赖（可失败） ==============#
# echo ">>> Installing system dependencies..."
# $SUDO_CMD apt update || echo "apt update failed"
# $SUDO_CMD apt install -y libcapstone-dev ninja-build || echo "apt install failed"

#================= [3] 回到项目根目录 =================#
echo ">>> Locating project root..."
project_dir=$(git rev-parse --show-toplevel)
cd "$project_dir"

#================= [4] 构建 wheel 包 ==================#
echo ">>> Building wheel package..."
python setup.py bdist_wheel

echo ">>> Built packages:"
ls -thl dist

#================= [5] 卸载旧版本 =====================#
if pip show module_logging > /dev/null 2>&1; then
    echo ">>> Uninstalling existing module_logging..."
    pip uninstall -y module_logging
else
    echo ">>> module_logging not installed, skipping uninstall."
fi

#================= [6] 安装新构建包 ===================#
echo ">>> Installing new package..."
WHL_FILE=$(ls dist/module_logging-*-cp*-cp*-linux_x86_64.whl | head -n 1)

if [ -f "$WHL_FILE" ]; then
    pip install --force-reinstall "$WHL_FILE"
    echo "✔️ Installed $WHL_FILE"
else
    echo "❌ No matching wheel found to install."
    exit 1
fi

echo "✅ Done!"
