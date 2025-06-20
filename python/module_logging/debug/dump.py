"""
this is a module for tensor debugging, to check the memory overlapping
"""

import threading
from functools import partial

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily

from ..configuration import get_config

OP_COUNTER = 0


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


class TensorInfoRecorder(TorchDispatchMode):
    """
    record the op input tensor infomation: element size, and data ptr
    """

    def __init__(self):
        """
        TensorInfoRecorder 类的初始化方法，用于初始化一些变量。
        """
        self.install_hook = False

        self.lock = threading.Lock()
        super().__init__()

    def __enter__(self):
        """
            在进入上下文管理器时，将被调用。此方法应该返回一个对象，该对象可以是任何类型，但通常是当前实例本身。
        这个方法的目的是为了让上下文管理器能够正确地处理和退出上下文管理器时需要执行的操作。

        Returns:
            Any: 当前实例本身，表示进入上下文管理器后的状态。
        """
        if not self.install_hook:
            self.install_hook = True
            cpp_extend = get_config("database", "cpp_extend")
            if cpp_extend == "True":
                from .. import Hook

                Hook.install_hook(Hook.HookType.kDUMP)
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

        if kwargs is None:
            kwargs = {}

        input_tensors = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_tensors.append(arg)
        for key in kwargs.keys():
            if isinstance(kwargs[key], torch.Tensor):
                input_tensors.append(kwargs[key])

        if cpp_extend == "True":
            from .. import Hook

            for i in range(len(input_tensors)):
                cur_tensor = input_tensors[i]
                Hook.record_tensor(
                    cur_tensor.data_ptr(),
                    cur_tensor.element_size() * cur_tensor.nelement(),
                )

        # call op
        output = op(*args, **kwargs)

        # relese thread lock
        self.lock.release()

        return output
