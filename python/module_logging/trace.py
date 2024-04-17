import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from torch.overrides import TorchFunctionMode, resolve_name
from contextlib import contextmanager
from . import config

cpp_extend = config.get_config("database", "cpp_extend")
if cpp_extend == "True":
    from . import Hook

class ProfilingLogger(TorchDispatchMode):
    """
    insert delimiters before and and after op execution
    """

    def __init__(self, model=None) -> None:
        super().__init__()
        self.start_time = Hook.get_current_time()

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        Hook.record_time("B", str(op), "aten op")

        # call op
        output = op(*args, **kwargs)

        Hook.record_time("E", str(op), "aten op")
        return output
