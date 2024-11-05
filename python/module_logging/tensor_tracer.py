import os
import sys
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.overrides import TorchFunctionMode, resolve_name
from contextlib import contextmanager
from . import config
from functools import partial
import prettytable as pt

cpp_extend = config.get_config("database", "cpp_extend")
if cpp_extend == "True":
    from . import Hook

MODULE_COUNTER = 0
print_rank = int(os.environ.get("PRINT_RANK", 0))


class Mode:
    MODULE = 0
    OP = 1


class TensorInfo:
    def __init__(self, name, tensor: torch.Tensor, mode=Mode.OP):
        self.name = name
        self.tensor = tensor
        self.mode = mode
        cpu_t = self.tensor.cpu()
        self.max = torch.max(cpu_t).item()
        self.min = torch.min(cpu_t).item()
        self.mean = torch.mean(cpu_t).item()
        self.std = torch.std(cpu_t).item()

    def get_tensor(self):
        return self.tensor

    def get_info(self):
        return self.max, self.min, self.mean, self.std

    def get_mode(self):
        return self.mode

    def try_release(self):
        ref_count = sys.getrefcount(self.tensor)
        if ref_count == 2:
            self.tensor = None
            return True
        return False

    def compare(self):
        cpu_t = self.tensor.detach().cpu().float()
        max_v = torch.max(cpu_t).item()
        min_v = torch.min(cpu_t).item()
        mean_v = torch.mean(cpu_t).item()
        std_v = torch.std(cpu_t).item()

        if (
            max_v != self.max
            or min_v != self.min
            or mean_v != self.mean
            or std_v != self.std
        ):
            table = pt.PrettyTable(
                [
                    "Tensor",
                    "Status",
                    "Max ",
                    "Min ",
                    "Mean",
                    "Std",
                ]
            )
            table.add_row([self.name, "old", self.max, self.min, self.mean, self.std])
            table.add_row([self.name, "new", max_v, min_v, mean_v, std_v])
            self.max = max_v
            self.min = min_v
            self.mean = mean_v
            self.std = std_v
            print(table)


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


class TensorTracer(TorchDispatchMode):
    """ """

    def __init__(self) -> None:
        super().__init__()
        self.trace_info = {}

    def __enter__(self):
        self._pt_impls = {}
        for k in TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        super().__enter__()

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        for k in TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        super().__exit__(exc_type, exc_value, traceback)

    def config(self, model):
        if isinstance(model, list):
            for module in model:
                for name, m in module.named_modules():
                    self._register_hook(m)
        elif isinstance(model, torch.nn.Module):
            for name, m in model.named_modules():
                self._register_hook(m)

    def trace(self, name, tensor, mode=Mode.OP):
        tensor_info = TensorInfo(name, tensor, mode)
        self.trace_info[name] = tensor_info

    def post_forward_hook_wrapper(self):
        def post_forward_hook(module, input, output):
            for t_name in self.trace_info.keys():
                t_info = self.trace_info[t_name]
                if t_info.get_mode() == Mode.MODULE:
                    t_info.compare()

        return post_forward_hook

    def post_backward_hook_wrapper(self):
        def post_backward_hook(module, input, output):
            for t_name in self.trace_info.keys():
                t_info = self.trace_info[t_name]
                if t_info.get_mode() == Mode.MODULE:
                    t_info.compare()

        return post_backward_hook

    def _register_hook(self, module):
        module.register_forward_hook(post_forward_hook_wrapper())
        module.register_full_backward_hook(post_backward_hook_wrapper())

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        print("[aten op name]: {}".format(str(op)))
        output = op(*args, **kwargs)
        # try to relase tensor, reduce memory peak
        for name in self.trace_info.keys():
            t_info = self.trace_info[name]
            if t_info.try_release():
                self.trace_info.pop(name)

        for name in self.trace_info.keys():
            t_info = self.trace_info[name]
            if t_info.get_mode() == Mode.OP:
                t_info.compare()
        return output


# tt = TensorTracer()
