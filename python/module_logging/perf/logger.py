import os
import threading
from contextlib import contextmanager
from functools import partial

import torch
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily

from ..configuration import get_config
from ..utils.logging import Logger

cpp_extend = get_config("database", "cpp_extend")
if cpp_extend == "True":
    pass

MODULE_COUNTER = 0
print_rank = int(os.environ.get("PRINT_RANK", 0))


def get_module_index():
    global MODULE_COUNTER
    MODULE_COUNTER += 1
    return MODULE_COUNTER


TENSOR_FUNCS_NO_DISPATCH = [
    # Can't convert Stream argument to Python object
    "record_stream"
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
        print("Initialize performance logger...")
        super().__init__()
        self.counter = 0
        self.rank = os.getpid()
        self.tid = threading.get_ident()
        self.lock = threading.Lock()
        enable_prof_env = os.environ.get("ENABLE_PROFILING", None)
        self.enable_profiling = False
        if enable_prof_env is not None:
            self.enable_profiling = enable_prof_env == "True" or enable_prof_env == "true"

        self.profiling_bw = profiling_bw
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
        if cpp_extend:
            from .. import Hook

            print("Install hook...")
            Hook.install_hook(Hook.HookType.kPROFILE)

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
        print("Enter performance logger...")
        if cpp_extend:
            from .. import Hook

            Hook.cuda_profiler_start()
        self._pt_impls = {}
        for k in TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        super().__enter__()
        # if cpp_extend:
        #     from .. import Hook
        # print("Performance logging entered at: {} ns".format(Hook.get_current_time()))

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        print("Exit performance logger...")
        if cpp_extend:
            from .. import Hook

            Hook.cuda_profiler_end()
        for k in TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        super().__exit__(exc_type, exc_value, traceback)
        # if cpp_extend:
        #     from .. import Hook
        # print("Performance logging exited at: {} ns".format(Hook.get_current_time()))

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

        return post_backward_hook

    def _register_hook(self, name, module):
        module.register_forward_pre_hook(self.pre_forward_hook_wrapper(name))
        module.register_forward_hook(self.post_forward_hook_wrapper(name))

        if self.profiling_bw:
            module.register_full_backward_pre_hook(self.pre_backward_hook_wrapper(name))
            module.register_full_backward_hook(self.post_backward_hook_wrapper(name))

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        self.lock.acquire()
        if kwargs is None:
            kwargs = {}
        if self.enable_profiling:
            torch.cuda.synchronize()
            #  insert pre-op delimiter
            print("[START_SYMBOL]: {} ns".format(str(op)), flush=True)
            # if cpp_extend:
            #     from .. import Hook
            #     print("{} start at: {}".format(str(op), Hook.get_current_time()))
            # for debug
            Logger.debug(
                "[START_SYMBOL]: {}, counter: {}, pid: {}, tid: {}".format(
                    str(op), self.counter, os.getpid(), threading.get_ident()
                )
            )
            # call op
            output = op(*args, **kwargs)
            torch.cuda.synchronize()
            # if cpp_extend:
            #     from .. import Hook
            #     print("{} end at: {} ns".format(str(op), Hook.get_current_time()))
            #  insert after-op delimiter
            print("[END_SYMBOL]: {}".format(str(op)), flush=True)
            # for debug
            Logger.debug(
                "[END_SYMBOL]: {}, counter: {}, pid: {}, tid: {}".format(
                    str(op), self.counter, os.getpid(), threading.get_ident()
                )
            )
            self.counter += 1
        else:
            return op(*args, **kwargs)
        self.lock.release()
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
