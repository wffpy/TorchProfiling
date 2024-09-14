import os
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.overrides import TorchFunctionMode, resolve_name
from contextlib import contextmanager
from . import config
from .utils import DistOpMonkeyPatch 
from functools import partial

cpp_extend = config.get_config("database", "cpp_extend")
if cpp_extend == "True":
    from . import Hook

MODULE_COUNTER = 0
print_rank = int(os.environ.get("PRINT_RANK", 0))


def get_module_index():
    global MODULE_COUNTER
    MODULE_COUNTER += 1
    return MODULE_COUNTER

TENSOR_FUNCS_NO_DISPATCH = [
    # Can't convert Stream argument to Python object
    'record_stream'
]

class TorchFuncMockNoDispatch:
    """
    Wraps a method to call it without the custom
    pytorch dispatcher
    """
    def __init__(self, pt_impl):
        self.pt_impl = pt_impl
    def __get__(self, obj, c):
        return partial(self, obj)
    def __call__(self, obj, *args, **kwargs):
        with _pop_mode_temporarily():
            return self.pt_impl(obj, *args, **kwargs)

class PerformanceLogger(TorchDispatchMode):
    """
    insert delimiters before and and after op execution
    """

    def __init__(self, model=None, profiling_bw=True) -> None:
        super().__init__()
        self.profiling_bw = profiling_bw
        # monkey patch for distributed op  
        self.monkey_patch = DistOpMonkeyPatch()
        # traverse modules and register forward and backward hooks for each
        if model:
            if isinstance(model, list):
                for module in model:
                    m_tuple = self.get_named_modules(module)
                    for name, m in m_tuple:
                        self._register_hook(name, m)
            elif isinstance(model, torch.nn.Module):
                m_tuple = self.get_named_modules(model)
                for name, m in m_tuple:
                    self._register_hook(name, m)

        # for gpu profilig with cpp extension, for xpu profiling is not necessary
        if config.cpp_extend():
            from . import Hook
            Hook.install_hook()

    def config(self, model=None, profiling_bw=True):
        if model:
            if isinstance(model, list):
                for module in model:
                    m_tuple = self.get_named_modules(module)
                    for name, m in m_tuple:
                        self._register_hook(name, m)
            elif isinstance(model, torch.nn.Module):
                m_tuple = self.get_named_modules(model)
                for name, m in m_tuple:
                    self._register_hook(name, m)

    def __enter__(self):
        if config.cpp_extend():
            Hook.cuda_profiler_start()
        self.monkey_patch.replace()
        self._pt_impls = {}
        for k in TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        if config.cpp_extend():
            Hook.cuda_profiler_end()
        self.monkey_patch.recover()
        for k in TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        super().__exit__(exc_type, exc_value, traceback)

    def get_named_modules(self, module: torch.nn.Module, prefix=""):
        stack = []
        m_name = module.__class__.__name__ if prefix == "" else prefix
        stack.append((m_name, module))
        acc_index = 0
        while acc_index < len(stack):
            f_name, f_m = stack[acc_index]
            child_modules = f_m.named_children()
            counter = 0
            for name, mod in child_modules:
                # construct module name
                if name == "":
                    name = "{}".format(counter)
                    counter += 1
                # store module name and module
                s_name = f_name + "#" + name
                s_m = mod
                stack.append((s_name, s_m))

            acc_index += 1
        return stack

    def pre_forward_hook_wrapper(self, name):
        def pre_forward_hook(module, input):
            torch.cuda.synchronize()
            print("[BEGIN FORWARD]: {}".format(name), flush=True)

        return pre_forward_hook

    def post_forward_hook_wrapper(self, name):
        def post_forward_hook(module, input, output):
            torch.cuda.synchronize()
            print("[END FORWARD]: {}".format(name), flush=True)
            # weights = module.state_dict()
            # print("weights number: {}".format(len(weights)))
            # print("weight keys: {}".format(weights.keys()))

        return post_forward_hook

    def pre_backward_hook_wrapper(self, name):
        def pre_backward_hook(module, input):
            torch.cuda.synchronize()
            print("[BEGIN BACKWARD]: {}_backward".format(name), flush=True)

        return pre_backward_hook

    def post_backward_hook_wrapper(self, name):
        def post_backward_hook(module, input, output):
            torch.cuda.synchronize()
            print("[END BACKWARD]: {}_backward".format(name), flush=True)
            # weights = module.state_dict()
            # print("weights number: {}".format(len(weights)))
            # print("weight keys: {}".format(weights.keys()))

        return post_backward_hook

    def _register_hook(self, name, module):
        module.register_forward_pre_hook(self.pre_forward_hook_wrapper(name))
        module.register_forward_hook(self.post_forward_hook_wrapper(name))

        if self.profiling_bw:
            module.register_full_backward_pre_hook(self.pre_backward_hook_wrapper(name))
            module.register_full_backward_hook(self.post_backward_hook_wrapper(name))

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        #  insert pre-op delimiter
        print("[START_SYMBOL]: {}".format(str(op)), flush=True)

        # call op
        torch.cuda.synchronize()
        output = op(*args, **kwargs)
        torch.cuda.synchronize()

        # if config.cpp_extend():
        #     Hook.cuda_profiler_flush()
        #  insert after-op delimiter
        print("[END_SYMBOL]: {}".format(str(op)), flush=True)
        return output


class TorchFunctionLog(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        # 打印 torch module接口
        print(f"[PYTORCH_FUNCTION_START]: {resolve_name(func)}", flush=True)
        ret = func(*args, **(kwargs or {}))
        print(f"[PYTORCH_FUNCTION_END]: {resolve_name(func)}", flush=True)
        # print(f"{resolve_name(func)}(*{args}, **{kwargs})", flush=True)
        return ret


@contextmanager
def TorchFunctionLogAndPerformanceLogger(model):
    with TorchFunctionLog():
        with PerformanceLogger(model):
            yield


@contextmanager
def combined_context(model=None):
    """
    with combined_context(model):
        train()
    """
    with TorchFunctionLogAndPerformanceLogger(model):
        yield
