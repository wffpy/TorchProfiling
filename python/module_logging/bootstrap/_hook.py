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
        # 不要在hook中向stdout打印任何语句
        module_hook(fullname, module)

        sys.meta_path.insert(0, finder)
        return module


sys.meta_path.insert(0, MetaPathFinder())


def module_hook(fullname, module):
    # print(f"fullname {fullname}")
    # print(f"module {module}")
    if fullname == "torch":
        module.autograd.backward = func_wrapper(module.autograd.backward)

        module.distributed.broadcast = func_torch_distributed_wrapper(
            module.distributed.broadcast
        )
        module.distributed.all_reduce = func_torch_distributed_wrapper(
            module.distributed.all_reduce
        )
        module.distributed.reduce = func_torch_distributed_wrapper(
            module.distributed.reduce
        )
        module.distributed.all_gather = func_torch_distributed_wrapper(
            module.distributed.all_gather
        )
        module.distributed.gather = func_torch_distributed_wrapper(
            module.distributed.gather
        )
        module.distributed.scatter = func_torch_distributed_wrapper(
            module.distributed.scatter
        )
        module.distributed.reduce_scatter = func_torch_distributed_wrapper(
            module.distributed.reduce_scatter
        )
        module.distributed.send = func_torch_distributed_wrapper(
            module.distributed.send
        )
        module.distributed.recv = func_torch_distributed_wrapper(
            module.distributed.recv
        )


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


def func_torch_distributed_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if callable(func):
            result = func(*args, **kwargs)
            if isinstance(args, tuple):
                args_info = ", ".join(
                    [
                        f"args_{idx} shape {arg.shape}, dtype {arg.dtype} "
                        for idx, arg in enumerate(args)
                    ]
                )
                print(
                    f"[PPROBE] torch.distributed.{func.__qualname__} {args_info}, kwargs {kwargs}"
                )
            else:
                print(
                    f"[PPROBE] torch.distributed.{func.__qualname__} args {args}, kwargs {kwargs} "
                )
            return result
        else:
            print(f"func:{func} is not callable")

    return wrapper
