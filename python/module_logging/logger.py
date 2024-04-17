import os
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from torch.overrides import TorchFunctionMode, resolve_name
from contextlib import contextmanager

MODULE_COUNTER = 0
print_rank = int(os.environ.get("PRINT_RANK", 0))


def get_module_index():
    global MODULE_COUNTER
    MODULE_COUNTER += 1
    return MODULE_COUNTER


class PerformanceLogger(TorchDispatchMode):
    """
    insert delimiters before and and after op execution
    """

    def __init__(self, model=None) -> None:
        super().__init__()
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
    # 判断是否处于分布式环境中
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == print_rank:
            with TorchFunctionLogAndPerformanceLogger(model):
                yield
        else:
            # 如果 rank 不是 0，则不执行任何操作
            yield
    else:
        # 默认单机
        with TorchFunctionLogAndPerformanceLogger(model):
            yield
