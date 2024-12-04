"""
this is a module for tensor debugging, to check the memory overlapping
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.overrides import TorchFunctionMode, resolve_name
from contextlib import contextmanager
from functools import partial
import traceback
import threading

OP_COUNTER = 0


class Mode:
    """
    the mode of tensor debugging
    """

    MODULE = 0
    OP = 1


class TensorInfo:
    """
    tensor information for debugging
    """

    def __init__(self, name, tensor=None, mode=Mode.OP):
        """
        init tensor information to compare
        """
        self.name = name
        self.tensor = tensor
        self.mode = mode
        cpu_t = self.tensor.cpu()
        self.max = torch.max(cpu_t).item()
        self.min = torch.min(cpu_t).item()
        self.mean = torch.mean(cpu_t).item()
        self.std = torch.std(cpu_t).item()

    def get_tensor(self):
        """
            获取当前张量。

        Returns:
            Tensor, 返回一个Tensor类型的对象，表示当前张量。
        """
        return self.tensor

    def get_info(self):
        """
            获取最大值、最小值、平均值和标准差信息。
        返回值：一个元组，包括最大值、最小值、平均值和标准差，分别对应于元组的第一到四个位置。

        Returns:
            tuple (float, float, float, float): 最大值、最小值、平均值和标准差信息。
        """
        return self.max, self.min, self.mean, self.std

    def get_mode(self):
        """
            获取当前模式，可能的值为：'train', 'val', 'test'。
        返回值（str）：当前模式。
        """
        return self.mode

    def try_release(self):
        """
            尝试释放 tensor，如果 tensor 的引用计数为 2，则释放。返回值是一个 bool，表示是否成功释放。
        如果 tensor 已经被其他对象引用，则不会释放。

        Args:
            None

        Returns:
            bool (bool): 如果成功释放，返回 True；否则返回 False。
        """
        ref_count = sys.getrefcount(self.tensor)
        if ref_count == 2:
            self.tensor = None
            return True
        return False

    def compare(self, skip_list=[]):
        """
            比较当前张量的最大值、最小值、平均值和标准差是否与上一次比较不同，如果不同则打印出来。

        Args:
            skip_list: tensor list to skip comparison

        Returns:
            无返回值，直接在控制台输出比较结果。
        """
        import prettytable as pt
        for skip in skip_list:
            if skip.data_ptr() == self.tensor.data_ptr():
                return False

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
            return True
        return False


def get_module_index():
    """
    获取模块索引，并自增。返回值为整数类型。

    Args:
        无参数。

    Returns:
        int (int): 当前模块的索引，从0开始计数。
    """
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
        """
            Initializes the class with a pointer to the underlying implementation.

        Args:
            pt_impl (cpointer): A pointer to the underlying implementation.
        """
        self.pt_impl = pt_impl

    def __get__(self, obj, c):
        """
            当属性被访问时，将返回一个partial函数，该函数将obj作为第一个参数。
        这样可以在调用partial函数时，自动传入obj作为第一个参数。

        Args:
            obj (Optional[Any]): 如果属性是实例属性，则obj表示实例对象；如果属性是类属性，则obj为None。默认值为None。
            c (Optional[type]): 类型提示，忽略此参数。默认值为None。

        Returns:
            Callable[[Any], Any]: 返回一个partial函数，将obj作为第一个参数。
        """
        return partial(self, obj)

    def __call__(self, obj, *args, **kwargs):
        """
            在一个临时的模式下调用 pt_impl，并返回结果。
        这个函数会自动把当前的模式压入栈，然后再调用 pt_impl，最后再将其弹出。

        Args:
            obj (Any): 传递给 pt_impl 的参数。
            args (Tuple[Any], optional): 可选参数，传递给 pt_impl 的其他参数。默认为空元组。
            kwargs (Dict[str, Any], optional): 可选参数，传递给 pt_impl 的关键字参数。默认为空字典。

        Returns:
            Any: pt_impl 的返回值。
        """
        with _pop_mode_temporarily():
            return self.pt_impl(obj, *args, **kwargs)


class TensorTracer(TorchDispatchMode):
    """
    A class that wraps PyTorch's dispatching mechanism and allows us to
    tracing the data changes of tensors.
    """

    def __init__(self):
        """
            初始化函数，用于初始化类的属性和方法。
        初始化完成后，类的属性包括：
            - trace_info (dict)：追踪信息字典，默认为空字典；
            - model_params (dict)：模型参数字典，默认为空字典；
            - trace_all (bool)：是否追踪所有输入，默认为False；
            - model (None)：当前使用的模型，默认为None；
            - backtrace (bool)：是否启用回溯，默认为False。
        """
        super().__init__()
        self.trace_info = {}
        self.model_params = {}
        self.trace_all = False
        self.model = None
        self.backtrace = False
        self.lock = threading.Lock()
        self.device = torch.device("cuda")

    def __enter__(self):
        """
            在进入上下文管理器时，将被调用。此方法应该返回一个对象，该对象可以是任何类型，但通常是当前实例本身。
        这个方法的目的是为了让上下文管理器能够正确地处理和退出上下文管理器时需要执行的操作。

        Returns:
            Any: 当前实例本身，表示进入上下文管理器后的状态。
        """
        self._pt_impls = {}
        for k in TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        super().__enter__()

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """
            重写 __exit__ 方法，在退出上下文管理器时恢复原始的 torch.Tensor 方法。
        如果有任何异常发生，则不会恢复原始的 torch.Tensor 方法。

        Args:
            exc_type (Optional[Type[BaseException]], optional): 异常类型（默认为 None）.
                Defaults to None.
            exc_value (Optional[BaseException], optional): 异常值（默认为 None）.
                Defaults to None.
            traceback (Optional[TracebackType], optional): Traceback 对象（默认为 None）.
                Defaults to None.

        Returns:
            None: 无返回值，只是用于恢复原始的 torch.Tensor 方法。
        """
        for k in TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        super().__exit__(exc_type, exc_value, traceback)

    def config(self, model, trace_all=False, trace_type=Mode.OP, backtrace=False):
        """
            配置追踪器，用于记录模型中的运算符。

        Args:
            model (Union[torch.nn.Module, List[torch.nn.Module]]): 需要追踪的模型或模型列表。
            trace_all (bool, optional, default=False): 是否对所有参数进行追踪，默认为 False。
            trace_type (Mode, optional, default=Mode.OP): 指定追踪类型，默认为 Mode.OP（运算符）。
            backtrace (bool, optional, default=False): 是否启用回溯功能，默认为 False。

        Returns:
            None.

        Raises:
            TypeError: 如果 `model` 不是 `torch.nn.Module` 或 `list` 类型，将会引发一个 TypeError 异常。
        """
        self.trace_all = trace_all
        self.model = model
        if isinstance(model, list):
            for module in model:
                for name, m in module.named_modules():
                    self._register_hook(m)
        elif isinstance(model, torch.nn.Module):
            for name, m in model.named_modules():
                self._register_hook(m)

        if self.trace_all and self.model is not None:
            if isinstance(self.model, torch.nn.Module):
                for name, param in model.named_parameters():
                    self.trace(name, param, trace_type)

            elif isinstance(self.model, list):
                for module in self.model:
                    for name, param in module.named_parameters():
                        self.trace(name, param, trace_type)
            else:
                raise TypeError(
                    "Expected `model` to be an instance of `torch.nn.Module` or `list`, but got {}.".format(
                        type(model)
                    )
                )

    def start(self):
        """
        启动服务器，开始监听客户端连接。
            该方法会被自动调用，无需手动调用。

            Args:
                None.

            Returns:
                None.

            Raises:
                None.
        """
        self.__enter__()

    def stop(self):
        """
            停止服务器，释放资源。
        该方法会在程序结束时自动调用，无需手动调用。
        """
        self.__exit__()

    def trace(self, name, tensor, mode=Mode.OP):
        """
            记录一个张量的信息，包括名称、张量本身以及追踪模式。
        如果该张量已经被追踪过，则不会重复添加。

        Args:
            name (str): 张量的名称。
            tensor (Tensor): 要追踪的张量。
            mode (Mode, optional): 追踪模式，默认为 Mode.OP（操作）。
                - Mode.OP：只追踪张量在计算图中的操作。
                - Mode.GRADIENT：只追踪张量对其他张量的梯度。
                - Mode.ALL：追踪张量在计算图中的所有操作和梯度。
                - Mode.NONE：不追踪张量。

        Returns:
            None.
        """
        tensor_info = TensorInfo(name, tensor, mode)
        self.trace_info[name] = tensor_info

    def trace_param(self, name, param, mode=Mode.OP):
        """
            记录参数的信息，包括名称、tensor、模式。
        该函数用于在追踪过程中记录参数的信息，以便后续使用。

        Args:
            name (str): 参数的名称。
            param (Tensor or Variable): 参数的tensor或variable对象。
            mode (Mode, optional): 参数所处的运算模式，默认为Mode.OP（运算）。可选项有Mode.FWD（前向）和Mode.BWD（反向）。
                Mode.FWD表示参数在前向传播阶段被使用，Mode.BWD表示参数在反向传播阶段被使用。

        Returns:
            None.

        Raises:
            None.
        """
        tensor_info = TensorInfo(name, param, mode)
        self.model_params[name] = tensor_info

    def post_forward_hook_wrapper(self):
        """
        返回一个post_forward_hook函数，用于在模型前向执行完成后进行追踪信息的比对。
            该函数会遍历所有已经记录的追踪信息，并调用每个追踪信息的compare方法进行比对。
            如果追踪信息处于模块模式，则会进行比对操作。

            Args:
                None

            Returns:
                function (post_forward_hook): 一个函数，接收三个参数（module、input、output），无返回值。
        """

        def post_forward_hook(module, input, output):
            for t_name in self.trace_info.keys():
                t_info = self.trace_info[t_name]
                if t_info.get_mode() == Mode.MODULE:
                    t_info.compare()

        return post_forward_hook

    def post_backward_hook_wrapper(self):
        """
            返回一个post_backward_hook函数，用于在反向传播完成后对模型进行追踪。
        该函数会遍历所有已经被追踪的模型，并调用其compare方法来比较输入和输出值是否相同。

        Args:
            无参数，不需要传入任何参数。

        Returns:
            function, post_backward_hook(module, input, output) -> None:
                一个函数，接收三个参数：module（torch.nn.Module类型），input（torch.Tensor类型），output（torch.Tensor类型），无返回值。
                该函数会遍历所有已经被追踪的模型，并调用其compare方法来比较输入和输出值是否相同。
        """

        def post_backward_hook(module, input, output):
            for t_name in self.trace_info.keys():
                t_info = self.trace_info[t_name]
                if t_info.get_mode() == Mode.MODULE:
                    t_info.compare()

        return post_backward_hook

    def _register_hook(self, module):
        """
            注册 hook，在 forward 和 backward 阶段执行相应的 wrapper。
        参数：
            module (nn.Module) - 需要注册 hook 的模型对象。
        返回值：
            无返回值，直接修改了 module 的属性。
        """
        module.register_forward_hook(self.post_forward_hook_wrapper())
        module.register_full_backward_hook(self.post_backward_hook_wrapper())

    def check_tensors(self, tensor_dict, op_name, skip_list=[]):
        """
        check tensors in tensor_dict and print the traceback if any tensor is modified

        Args:
            tensor_dict (Dict[str, TensorInfo]): 包含所有张量信息的字典，其中key为张量名称，value为TensorInfo对象。
            op_name (str): current operator name。
            skip_list (List[Tensor], optional): the tenosrs to skip checking, default [].

        Returns:
            None.
        """
        for name in tensor_dict.keys():
            t_info = tensor_dict[name]
            if t_info.get_mode() == Mode.OP:
                modified = t_info.compare(skip_list)
                if modified:
                    print("[current op]: {}".format(op_name))
                    if self.backtrace:
                        print("".join(traceback.format_stack()))

    def release_tensor(self):
        """
        try to release the temporary tensors which is out of scope
        condition:
            when a tensor's reference count is 2, it can be released
        """
        release_list = []
        for name in self.trace_info.keys():
            t_info = self.trace_info[name]
            if t_info.try_release():
                release_list.append(name)

        for name in release_list:
            self.trace_info.pop(name)

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        """
        Override the default torch dispatch method to handle custom ops.

        Args:
            op (function): The operation to be performed.
            types (Tuple[Type]): A tuple of input types.
            args (Optional[Tuple], optional): Arguments for the op. Defaults to ().
            kwargs (Optional[Dict], optional): Keyword arguments for the op. Defaults to None.

        Returns:
            Any: The result of the operation.

        Raises:
            TypeError: If the input types are not compatible with the op.
        """
        # acquire thread lock
        self.lock.acquire()

        global OP_COUNTER
        if kwargs is None:
            kwargs = {}

        # try to relase tensor, reduce memory peak
        # this action should do before op is called
        # model params will not be released, so no need to check it
        self.release_tensor()

        # call op
        output = op(*args, **kwargs)

        # collect all input tensors
        input_tensors = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_tensors.append(arg)
        for key in kwargs.keys():
            if isinstance(kwargs[key], torch.Tensor):
                input_tensors.append(kwargs[key])

        # collect all output tensors
        output_tensors = []
        if isinstance(output, torch.Tensor):
            output_tensors.append(output)
        elif isinstance(output, tuple):
            for item in output:
                if isinstance(item, torch.Tensor):
                    output_tensors.append(item)

        skip_list = []
        trace_list = []
        # inplaced tensor no need to trace in current op
        for output_tensor in output_tensors:
            # just check cuda tensor
            if output_tensor.is_cuda:
                for in_tensor in input_tensors:
                    if in_tensor.data_ptr() == output_tensor.data_ptr():
                        skip_list.append(output_tensor)
                    else:
                        trace_list.append(output_tensor)

        op_name = str(op)

        self.check_tensors(self.trace_info, op_name, skip_list)

        if self.trace_all:
            self.check_tensors(self.model_params, op_name, skip_list)

            # trace not inplaced output tensor
            unique_op_name = "{}_{}".format(op_name, OP_COUNTER)

            for trace_tensor in trace_list:
                self.trace(unique_op_name, trace_tensor, Mode.OP)

        OP_COUNTER += 1

        # relese thread lock
        self.lock.release()

        return output
