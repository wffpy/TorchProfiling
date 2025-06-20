import torch
import ctypes
import pybind11
from ..configuration import get_config, cpp_extend
enable_cpp_extend = cpp_extend()

def py_xpu_launch_async(nclusters, ncores, stream):
    is_cap = torch.cuda.is_current_stream_capturing()
    if is_cap: 
        print("launch stream: {}".format(stream))
    curr_stream = torch.cuda.current_stream() 
    print("curr_stream: {}".format(curr_stream))

    print("py_xpu_launch_async")
    func_name = "xpu_launch_config"
    origin_func = hook.get_origin_func(func_name)
    origin_func_ptr = ctypes.cast(origin_func, ctypes.c_void_p).value
    py_xpu_launch_async_origin = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)(origin_func_ptr)
    ret = py_xpu_launch_async_origin(nclusters, ncores, stream)
    return ret


class GraphDebug:
    def __init__(self):
        print("Enable Graph Debug")
        py_func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)(py_xpu_launch_async)
        capsule = ctypes.pythonapi.PyCapsule_New(ctypes.cast(py_func, ctypes.c_void_p).value, None, None)
        if enable_cpp_extend:
            from .. import Hook
            Hook.register_got_hook(hook.HookType.kDUBUG, "xpu_launch_config", capsule)
            Hook.install_hook(hook.HookType.kDUBUG)
