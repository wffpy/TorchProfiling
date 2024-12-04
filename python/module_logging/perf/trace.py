import torch
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from . import config
from functools import partial
import os

rank = os.getenv("RANK", "0")

cpp_extend = config.get_config("database", "cpp_extend")
if cpp_extend == "True":
    from .. import Hook


from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    _DistributedBackendOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    PrefixStore,
    ProcessGroup,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    DebugLevel,
    get_debug_level,
    Work,
)

origin_all_reduce = None
origin_broadcast = None
origin_barrier = None
origin__all_gather_base = None
origin__reduce_scatter_base = None
origin_all_gather = None
origin_send = None
origin_recv = None


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner


def mock_all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    # print("[DIST START_SYMBOL]: torch.distributed.all_reduce", flush=True)
    Hook.record_time("B", "torch.distributed.all_reduce", "aten op")
    # bytest = tensor.numel() * tensor.element_size()
    # print("[DIST BYTES]:  {} bytes".format(bytest), flush=True)
    ret = origin_all_reduce(tensor, op, group, async_op)
    # print("[DIST END_SYMBOL]: torch.distributed.all_reduce", flush=True)
    Hook.record_time("E", "torch.distributed.all_reduce", "aten op")
    return ret


def mock_broadcast(tensor, src, group=None, async_op=False):
    # print("[DIST START_SYMBOL]: torch.distributed.broadcast", flush=True)
    Hook.record_time("B", "torch.distributed.broadcast", "aten op")
    # bytest = tensor.numel() * tensor.element_size()
    # print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
    ret = origin_broadcast(tensor, src, group, async_op)
    # print("[DIST END_SYMBOL]: torch.distributed.broadcast", flush=True)
    Hook.record_time("E", "torch.distributed.broadcast", "aten op")
    return ret


def mock_barrier(group=None, async_op=False, device_ids=None):
    # print("[DIST START_SYMBOL]: torch.distributed.barrier", flush=True)
    Hook.record_time("B", "torch.distributed.barrier", "aten op")
    ret = origin_barrier(group, async_op, device_ids)
    Hook.record_time("E", "torch.distributed.barrier", "aten op")
    # print("[DIST END_SYMBOL]: torch.distributed.barrier", flush=True)
    return ret


# def mock_all_gather(tensor_list, tensor, group=None, async_op=False):
#     return
def mock_all_gather(tensor_list, tensor, group=None, async_op=False):
    # print("[DIST START_SYMBOL]: torch.distributed.all_gather", flush=True)
    # bytest = tensor.numel() * tensor.element_size()
    Hook.record_time("B", "torch.distributed.all_gather", "aten op")
    # print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
    ret = origin_all_gather(tensor_list, tensor, group, async_op)
    Hook.record_time("E", "torch.distributed.all_gather", "aten op")
    # print("[DIST END_SYMBOL]: torch.distributed.all_gather", flush=True)
    return ret


def mock__all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
    # print("[DIST START_SYMBOL]: torch.distributed._all_gather_base", flush=True)
    # bytest = output_tensor.numel() * output_tensor.element_size()
    # print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
    Hook.record_time("B", "torch.distributed._all_gather_base", "aten op")
    ret = origin__all_gather_base(output_tensor, input_tensor, group, async_op)
    # print("[DIST END_SYMBOL]: torch.distributed._all_gather_base", flush=True)
    Hook.record_time("E", "torch.distributed._all_gather_base", "aten op")
    return ret


def mock__reduce_scatter_base(
    output_tensor, input, op=ReduceOp.SUM, group=None, async_op=False
):
    # print("[DIST START_SYMBOL]: torch.distributed._reduce_scatter_base", flush=True)
    # bytest = output_tensor.numel() * output_tensor.element_size()
    # print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
    Hook.record_time("B", "torch.distributed._reduce_scatter_base", "aten op")
    ret = origin__reduce_scatter_base(output_tensor, input, op, group, async_op)
    # print("[DIST END_SYMBOL]: torch.distributed._reduce_scatter_base", flush=True)
    Hook.record_time("E", "torch.distributed._reduce_scatter_base", "aten op")
    return ret


def mock_send(
    tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0
) -> None:
    # print("[DIST START_SYMBOL]: torch.distributed.send", flush=True)
    # bytest = tensor.numel() * tensor.element_size()
    # print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
    Hook.record_time("B", "torch.distributed.send", "aten op")
    ret = origin_send(tensor, dst, group, tag)
    # print("[DIST END_SYMBOL]: torch.distributed.send", flush=True)
    Hook.record_time("E", "torch.distributed.send", "aten op")
    return ret


def mock_recv(
    tensor: torch.Tensor,
    src: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
    tag: int = 0,
) -> None:
    # print("[DIST START_SYMBOL]: torch.distributed.recv", flush=True)
    # bytest = tensor.numel() * tensor.element_size()
    # print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
    Hook.record_time("B", "torch.distributed.recv", "aten op")
    ret = origin_recv(tensor, src, group, tag)
    # print("[DIST END_SYMBOL]: torch.distributed.recv", flush=True)
    Hook.record_time("E", "torch.distributed.recv", "aten op")
    return ret


@singleton
class DistOpRecordMonkeyPatch(object):

    def __init__(self):
        global origin_all_reduce
        global origin_broadcast
        global origin_barrier
        global origin__all_gather_base
        global origin__reduce_scatter_base
        global origin_all_gather
        global origin_send
        global origin_recv

        origin_all_reduce = torch.distributed.all_reduce
        origin_broadcast = torch.distributed.broadcast
        origin_barrier = torch.distributed.barrier
        origin__all_gather_base = torch.distributed._all_gather_base
        origin__reduce_scatter_base = torch.distributed._reduce_scatter_base
        origin_all_gather = torch.distributed.all_gather
        origin_send = torch.distributed.send
        origin_recv = torch.distributed.recv

    def replace(self):
        """
        use monkey patch to replace the original function
        """
        torch.distributed.all_reduce = mock_all_reduce
        torch.distributed.broadcast = mock_broadcast
        torch.distributed.barrier = mock_barrier
        torch.distributed._all_gather_base = mock__all_gather_base
        torch.distributed._reduce_scatter_base = mock__reduce_scatter_base
        torch.distributed.all_gather = mock_all_gather
        torch.distributed.send = mock_send
        torch.distributed.recv = mock_recv

    def recover(self):
        """
        recover the original function
        """
        torch.distributed.all_reduce = origin_all_reduce
        torch.distributed.broadcast = origin_broadcast
        torch.distributed.barrier = origin_barrier
        torch.distributed._all_gather_base = origin__all_gather_base
        torch.distributed._reduce_scatter_base = origin__reduce_scatter_base
        torch.distributed.all_gather = origin_all_gather
        torch.distributed.send = origin_send
        torch.distributed.recv = origin_recv


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


class Tracer(TorchDispatchMode):
    """
    insert delimiters before and and after op execution
    """

    def __init__(
        self,
        model=None,
        path=None,
        profiling_bw=False,
        print_module_info=True,
        ranks=None,
    ) -> None:
        """
        model: nn.Module or nn.Module list to be traced
        path: path to save profiling data
        profling_bw: whether to profile backward pass, for some specific case, profiling backward pass
                     will lead to following error: RuntimeError:
                     "Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace.
                     This view was created inside a custom Function (or because an input was returned as-is)
                     and the autograd logic to handle view+inplace would override the custom backward associated with the custom Function, leading to incorrect gradients. This behavior is forbidden.
                     You can fix this by cloning the output of the custom Function."
        print_module_info: whether to print module info: e.g. BEGIN FORWARD: {}_froward, END FORWARD: {}_froward, BEGIN BACKWARD: {}_backward, END BACKWARD: {}_backward
        """
        super().__init__()
        self.profiling_backward = profiling_bw
        self.print_module_info = print_module_info
        self.monkey_patch = DistOpRecordMonkeyPatch()
        self.ranks = ranks
        if self.ranks and rank and int(rank) not in self.ranks:
            return

        # install hooks for some runtime api / fprintf to record time
        Hook.install_hook(Hook.HookType.kPROFILE)

        # enable recorder to record the profiling logs and writo file
        if self.print_module_info:
            Hook.enable_recorder()
            log_path = "/tmp/logs/{}.log".format(rank)
            Hook.set_log_record_path(log_path)

        # enable timer recording
        Hook.enable_profiling()

        # set path to record profiling data
        if path is None:
            Hook.set_timer_record_path("/tmp/profiling.json")
        else:
            Hook.set_timer_record_path(path)

        if model is None:
            return
        else:
            if isinstance(model, list):
                for module in model:
                    m_tuple = self.get_named_modules(module)
                    for name, m, l in m_tuple:
                        self._register_hook(name, m, l)
            elif isinstance(model, torch.nn.Module):
                m_tuple = self.get_named_modules(model)
                for name, m, l in m_tuple:
                    self._register_hook(name, m, l)

    def __enter__(self):
        self.monkey_patch.replace()
        self._pt_impls = {}
        for k in TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.monkey_patch.recover()
        for k in TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        super().__exit__(exc_type, exc_value, traceback)
        Hook.write_to_file()
        Hook.close_recorder()

    def get_named_modules(self, module: torch.nn.Module, prefix=""):
        stack = []
        level = 0
        max_level = 0
        m_name = module.__class__.__name__ if prefix == "" else prefix
        stack.append((m_name, module, level))
        acc_index = 0
        while acc_index < len(stack):
            f_name, f_m, l = stack[acc_index]
            child_modules = f_m.named_children()
            max_level = max(max_level, l)
            counter = 0
            for name, mod in child_modules:
                # construct module name
                if name == "":
                    name = "{}".format(counter)
                    counter += 1
                # store module name and module
                s_name = f_name + "#" + name
                s_m = mod
                stack.append((s_name, s_m, l + 1))

            acc_index += 1
        return stack

    def pre_forward_hook_wrapper(self, name, level):
        def pre_forward_hook(module, input):
            level_name = "Module L{}".format(level)
            if self.print_module_info:
                log_str = "[BEGIN FORWARD]: {}".format(name)
                Hook.record_log(log_str)
            Hook.record_time("B", str(name), level_name)

        return pre_forward_hook

    def post_forward_hook_wrapper(self, name, level):
        def post_forward_hook(module, input, output):
            level_name = "Module L{}".format(level)
            if self.print_module_info:
                log_str = "[END FORWARD]: {}".format(name)
                Hook.record_log(log_str)
            Hook.record_time("E", str(name), level_name)

        return post_forward_hook

    def pre_backward_hook_wrapper(self, name, level):
        def pre_backward_hook(module, input):
            level_name = "Module L{}".format(level)
            if self.print_module_info:
                log_str = "[BEGIN BACKWARD]: {}_backward".format(name)
                Hook.record_log(log_str)
            Hook.record_time("B", str(name), level_name)

        return pre_backward_hook

    def post_backward_hook_wrapper(self, name, level):
        def post_backward_hook(module, input, output):
            level_name = "Module L{}".format(level)
            if self.print_module_info:
                log_str = "[END BACKWARD]: {}_backward".format(name)
                Hook.record_log(log_str)
            Hook.record_time("E", str(name), level_name)

        return post_backward_hook

    def _register_hook(self, name, module, level):
        module.register_forward_pre_hook(self.pre_forward_hook_wrapper(name, level))
        module.register_forward_hook(self.post_forward_hook_wrapper(name, level))

        if self.profiling_backward:
            module.register_full_backward_pre_hook(
                self.pre_backward_hook_wrapper(name, level)
            )
            module.register_full_backward_hook(
                self.post_backward_hook_wrapper(name, level)
            )

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if self.print_module_info:
            Hook.record_log("[START_SYMBOL]: {}".format(str(op)))
        Hook.record_time("B", str(op), "aten op")

        # call op
        output = op(*args, **kwargs)

        Hook.record_time("E", str(op), "aten op")
        if self.print_module_info:
            Hook.record_log("[END_SYMBOL]: {}".format(str(op)))
        return output
