import functools
import importlib
import sys
import time
import os

_hook_modules = {"torch"}
times = 0


class MetaPathFinder:
    def find_module(self, fullname, path=None):
        if fullname in _hook_modules:
            return MetaPathLoader()


class MetaPathLoader:
    def load_module(self, fullname):
        # print("load_module {}".format(fullname))
        # ``sys.modules`` 中保存的是已经导入过的 module
        if fullname in sys.modules:
            return sys.modules[fullname]
        print("not already imported")

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
    print("Hook Distributed Ops")
    if fullname == "torch":
        # module.autograd.backward = func_wrapper(module.autograd.backward)

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
        # module.distributed._allgather_base= func_torch_distributed_wrapper(
        #     module.distributed._allgather_base
        # )
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

class INFOTYPE:
    BEFORE = 1
    AFTER = 2
    DATA = 3

def get_param(args, kwargs, position: int, param_name: str):
    if len(args) > position:
        return args[position]
    elif param_name in kwargs:
        return kwargs[param_name]
    else:
        assert False, "No such parameter: {}".format(param_name)
        return None

def gen_bytes_str(tensor):
    bytes = tensor.numel() * tensor.element_size()
    return "[DIST BYTES]:  {} bytes".format(bytes) 

class DistInfoGenerator(object):
    @staticmethod
    def gen_broadcast(args, kwargs):
        tensor = get_param(args, kwargs, 0, "tensor")
        if tensor is not None:
            return gen_bytes_str(tensor)
        return None

    @staticmethod
    def gen_all_reduce(args, kwargs):
        tensor = get_param(args, kwargs, 0, "tensor")
        if tensor is not None:
            return gen_bytes_str(tensor)
        return None

    @staticmethod
    def gen_barrier(args, kwargs):
        return None

    @staticmethod
    def gen_all_gather(args, kwargs):
        tensor = get_param(args, kwargs, 1, "tensor")
        if tensor is not None:
            return gen_bytes_str(tensor)
        return None

    @staticmethod
    def gen__allgather_base(args, kwargs):
        tensor = get_param(args, kwargs, 0, "output_tensor")
        if tensor is not None:
            return gen_bytes_str(tensor)
        return None

    @staticmethod
    def gen_reduce_scatter(args, kwargs):
        tensor = get_param(args, kwargs, 0, "output_tensor")
        if tensor is not None:
            return gen_bytes_str(tensor)
        return None


    @staticmethod
    def gen_send(args, kwargs):
        tensor = get_param(args, kwargs, 0, "tensor")
        if tensor is not None:
            return gen_bytes_str(tensor)
        return None

    @staticmethod
    def gen_recv(args, kwargs):
        tensor = get_param(args, kwargs, 0, "tensor")
        if tensor is not None:
            return gen_bytes_str(tensor)
        return None

# op_name: all_reduce_
def print_dist_op_bytes_str(op_name, args, kwargs):
    gen_func_name = "gen_{}".format(op_name)
    if hasattr(DistInfoGenerator, gen_func_name):
        func = getattr(DistInfoGenerator, gen_func_name)
        out = func(args, kwargs)
        if out is not None:
            print(out)
    else:
        # assert False, "No such function: {}".format(gen_func_name)
        print("No such function: {}".format(gen_func_name))

def enable_profiling():
    enable_prof_env = os.environ.get('ENABLE_PROFILING', None)
    if enable_prof_env is not None :
        if enable_prof_env == 'true' or enable_prof_env == 'True':
            return True
    return False

def func_torch_distributed_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if callable(func):
            if enable_profiling():
                import torch
                func_name = func.__name__
                torch.cuda.synchronize()
                print("[DIST START_SYMBOL]: {}".format(func.__name__))
                print_dist_op_bytes_str(func_name, args, kwargs)
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                print("[DIST END_SYMBOL]: {}".format(func.__name__))
                return result
            else:
                result = func(*args, **kwargs)
        else:
            assert False, "func:{} is not callable".format(func)

    return wrapper
