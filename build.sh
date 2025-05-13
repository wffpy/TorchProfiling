set -e  # 遇到错误立即退出
set -o pipefail

PACKAGE_NAME="module_logging"

echo "========= [1/5] Clean dist/ directory ========="
rm -rf dist/*
echo "✔️ Cleaned dist/"

echo "========= [2/5] Build wheel package =========="
python setup.py sdist bdist_wheel > /dev/null
echo "✔️ Build complete"

echo "========= [3/5] Uninstall existing package ===="
pip show $PACKAGE_NAME > /dev/null 2>&1 && {
    pip uninstall -y $PACKAGE_NAME
    echo "✔️ Uninstalled existing $PACKAGE_NAME"
} || {
    echo "ℹ️ $PACKAGE_NAME not installed, skipping uninstall"
}

echo "========= [4/5] Install newly built package ==="
pip install --force-reinstall dist/*.whl
echo "✔️ Install complete"

echo "========= [5/5] Done =========================="
