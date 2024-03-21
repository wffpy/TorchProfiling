echo "=========begin build whl package=============="
python setup.py sdist bdist_wheel

echo "=========Try to uninstall origin package=============="

echo -e "Y" | pip uninstall module_logging

echo "=========install new package=============="
pip install dist/module_logging-1.0.0-cp38-cp38-linux_x86_64.whl

echo "Done !"
# bcecmd bos cp dist/module_logging-1.0.0-py3-none-any.whl  bos:/klx-pytorch-work-bd/tmp/fangfei/
