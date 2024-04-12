import functools
import importlib
import sys
import time

_hook_modules = {"torch"}
times = 0


class MetaPathFinder:
    def find_module(self, fullname, path=None):
        # print("find_module {}".format(fullname))
        if fullname in _hook_modules:
            return MetaPathLoader()


class MetaPathLoader:
    def load_module(self, fullname):
        # print("load_module {}".format(fullname))
        # ``sys.modules`` 中保存的是已经导入过的 module
        if fullname in sys.modules:
            return sys.modules[fullname]

        # 先从 sys.meta_path 中删除自定义的 finder
        # 防止下面执行 import_module 的时候再次触发此 finder
        # 从而出现递归调用的问题
        finder = sys.meta_path.pop(0)
        # 导入 module
        module = importlib.import_module(fullname)
        print(" --------------- ", fullname, " ================== ", module)
        module_hook(fullname, module)

        sys.meta_path.insert(0, finder)
        return module


sys.meta_path.insert(0, MetaPathFinder())


def module_hook(fullname, module):
    print(f"fullname {fullname}")
    print(f"module {module}")
    if fullname == "torch":
        # TODO 以下backward是等效的, 可能极少数特殊情况下会有重复计数的情况
        module.Tensor.backward = func_wrapper(module.Tensor.backward)
        module.autograd.backward = func_wrapper(module.autograd.backward)


def func_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global times
        print(f">>>>>>>>>>>> {args}")
        print(f"start func {func}")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        times += 1
        print("spent {}s, count==>{}".format(end - start, times))
        return result

    return wrapper
