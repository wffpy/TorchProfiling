import os
import time
import threading
from contextlib import contextmanager
from functools import partial

import torch
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily

from ..configuration import get_config, cpp_extend
from ..utils.logging import Logger

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


class CpuPerformanceLogger(TorchDispatchMode):
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
            self.enable_profiling = (
                enable_prof_env == "True" or enable_prof_env == "true"
            )

        self.profiling_bw = profiling_bw


    def config(self, model=None, profiling_bw=True):
        pass

    def __enter__(self):
        print("Enter performance logger...")
        self._pt_impls = {}
        for k in TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        super().__enter__()

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        print("Exit performance logger...")
        for k in TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        super().__exit__(exc_type, exc_value, traceback)

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        output = None
        if kwargs is None:
            kwargs = {}

        self.lock.acquire()
        start = time.time_ns()
        # call op
        output = op(*args, **kwargs)
        end = time.time_ns()
        cpu_time = end - start
        print("[OP-NAME]: {}: {}".format(str(op), cpu_time), flush=True)
        self.lock.release()
        return output
