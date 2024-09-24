import os
import torch
from torch._C._distributed_c10d import ReduceOp
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
    Work
)
# from logging import Logger

# class GlobalFlags:
#     # 声明一个类变量作为全局 flag
#     flag = False

#     @classmethod
#     def set_flag(cls, value):
#         """设置全局 flag 的值"""
#         cls.flag = value

#     @classmethod
#     def get_flag(cls):
#         """获取全局 flag 的值"""
#         return cls.flag

# # 设置全局 flag
# GlobalFlags.set_flag(False)

# origin_all_reduce = None
# origin_broadcast = None
# origin_barrier = None
# origin__all_gather_base = None
# origin__reduce_scatter_base = None
# origin_all_gather = None
# origin_send = None
# origin_recv = None

# def singleton(cls):
#     _instance = {}

#     def inner():
#         if cls not in _instance:
#             _instance[cls] = cls()
#         return _instance[cls]
#     return inner
        
# def mock_all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
#     if GlobalFlags.get_flag() == True:
#         torch.cuda.synchronize()
#         print("[DIST START_SYMBOL]: torch.distributed.all_reduce", flush=True)
#         bytest = tensor.numel() * tensor.element_size()
#         print("[DIST BYTES]:  {} bytes".format(bytest), flush=True)
#         ret = origin_all_reduce(tensor, op, group, async_op)
#         torch.cuda.synchronize()
#         print("[DIST END_SYMBOL]: torch.distributed.all_reduce", flush=True)
#         return ret
#     else:
#         return origin_all_reduce(tensor, op, group, async_op)

# def mock_broadcast(tensor, src, group=None, async_op=False):
#     if GlobalFlags.get_flag() == True:
#         torch.cuda.synchronize()
#         print("[DIST START_SYMBOL]: torch.distributed.broadcast", flush=True)
#         bytest = tensor.numel() * tensor.element_size()
#         print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
#         ret = origin_broadcast(tensor, src, group, async_op)
#         torch.cuda.synchronize()
#         print("[DIST END_SYMBOL]: torch.distributed.broadcast", flush=True)
#         return ret
#     else:
#         return origin_broadcast(tensor, src, group, async_op)

# def mock_barrier(group=None, async_op=False, device_ids=None):
#     if GlobalFlags.get_flag() == True:
#         torch.cuda.synchronize()
#         print("[DIST START_SYMBOL]: torch.distributed.barrier", flush=True)
#         ret = origin_barrier(group, async_op, device_ids)
#         torch.cuda.synchronize()
#         print("[DIST END_SYMBOL]: torch.distributed.barrier", flush=True)
#         return ret
#     else:
#         return origin_barrier(group, async_op, device_ids)

# # def mock_all_gather(tensor_list, tensor, group=None, async_op=False):
# #     return
# def mock_all_gather(tensor_list, tensor, group=None, async_op=False):
#     if GlobalFlags.get_flag() == True:
#         torch.cuda.synchronize()
#         print("[DIST START_SYMBOL]: torch.distributed.all_gather", flush=True)
#         bytest = tensor.numel() * tensor.element_size()
#         print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
#         ret = origin_all_gather(tensor_list, tensor, group, async_op)
#         torch.cuda.synchronize()
#         print("[DIST END_SYMBOL]: torch.distributed.all_gather", flush=True)
#         return ret
#     else:
#         return origin_all_gather(tensor_list, tensor, group, async_op)

# def mock__all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
#     if GlobalFlags.get_flag() == True:
#         torch.cuda.synchronize()
#         print("[DIST START_SYMBOL]: torch.distributed._all_gather_base", flush=True)
#         bytest = output_tensor.numel() * output_tensor.element_size()
#         print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
#         ret = origin__all_gather_base(output_tensor, input_tensor, group, async_op)
#         torch.cuda.synchronize()
#         print("[DIST END_SYMBOL]: torch.distributed._all_gather_base", flush=True)
#         return ret
#     else:
#         return origin__all_gather_base(output_tensor, input_tensor, group, async_op)

# def mock__reduce_scatter_base(
#     output_tensor, input, op=ReduceOp.SUM, group=None, async_op=False
# ):
#     if GlobalFlags.get_flag() == True:
#         torch.cuda.synchronize()
#         print("[DIST START_SYMBOL]: torch.distributed._reduce_scatter_base", flush=True)
#         bytest = output_tensor.numel() * output_tensor.element_size()
#         print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
#         ret = origin__reduce_scatter_base(output_tensor, input, op, group, async_op)
#         torch.cuda.synchronize()
#         print("[DIST END_SYMBOL]: torch.distributed._reduce_scatter_base", flush=True)
#         return ret

# def mock_send(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0) -> None:
#     if GlobalFlags.get_flag() == True:
#         torch.cuda.synchronize()
#         print("[DIST START_SYMBOL]: torch.distributed.send", flush=True)
#         bytest = tensor.numel() * tensor.element_size()
#         print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
#         ret = origin_send(tensor, dst, group, tag)
#         torch.cuda.synchronize()
#         print("[DIST END_SYMBOL]: torch.distributed.send", flush=True)
#         return ret 
#     else:
#         return origin_send(tensor, dst, group, tag)

# def mock_recv(tensor: torch.Tensor, src: Optional[int] = None, group: Optional[ProcessGroup] = None, tag: int = 0) -> None:
#     if GlobalFlags.get_flag() == True:
#         torch.cuda.synchronize()
#         print("[DIST START_SYMBOL]: torch.distributed.recv", flush=True)
#         bytest = tensor.numel() * tensor.element_size()
#         print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
#         ret = origin_recv(tensor, src, group, tag)
#         torch.cuda.synchronize()
#         print("[DIST END_SYMBOL]: torch.distributed.recv", flush=True)
#         return  ret
#     else:
#         return origin_recv(tensor, src, group, tag)

# @singleton
# class DistOpMonkeyPatch(object):

#     def __init__(self):
#         global origin_all_reduce
#         global origin_broadcast
#         global origin_barrier
#         global origin__all_gather_base
#         global origin__reduce_scatter_base
#         global origin_all_gather
#         global origin_send
#         global origin_recv

#         origin_all_reduce = torch.distributed.all_reduce
#         origin_broadcast = torch.distributed.broadcast
#         origin_barrier = torch.distributed.barrier
#         origin__all_gather_base = torch.distributed._all_gather_base
#         origin__reduce_scatter_base = torch.distributed._reduce_scatter_base
#         origin_all_gather = torch.distributed.all_gather
#         origin_send = torch.distributed.send
#         origin_recv = torch.distributed.recv

#     def replace(self):
#         '''
#         use monkey patch to replace the original function
#         '''
#         torch.distributed.all_reduce = mock_all_reduce
#         torch.distributed.broadcast = mock_broadcast
#         torch.distributed.barrier = mock_barrier
#         torch.distributed._all_gather_base = mock__all_gather_base
#         torch.distributed._reduce_scatter_base = mock__reduce_scatter_base
#         torch.distributed.all_gather = mock_all_gather
#         torch.distributed.send = mock_send
#         torch.distributed.recv = mock_recv

#     def recover(self):
#         '''
#         recover the original function
#         '''
#         torch.distributed.all_reduce = origin_all_reduce 
#         torch.distributed.broadcast = origin_broadcast 
#         torch.distributed.barrier = origin_barrier 
#         torch.distributed._all_gather_base = origin__all_gather_base
#         torch.distributed._reduce_scatter_base = origin__reduce_scatter_base
#         torch.distributed.all_gather = origin_all_gather
#         torch.distributed.send = origin_send
#         torch.distributed.recv = origin_recv

# # inpu texample: c10d.all_reduce_.default
# def is_dist_op(name):
#     if name.startswith('c10d'):
#         return True

# # input example: c10d.all_reduce_.default
# def get_dist_op_name(name):
#     # name string format: c10d.all_reduce_.default
#     assert is_dist_op(name), "{} is not a dist op".format(name)
#     # example case: ['c10d', 'all_reduce_', 'default']
#     str_list = name.split('.')
#     return str_list[1]

# class INFOTYPE:
#     BEFORE = 1
#     AFTER = 2
#     DATA = 3

# def get_param(args, kwargs, position: int, param_name: str):
#     if len(args) > position:
#         return args[position]
#     elif param_name in kwargs:
#         return kwargs[param_name]
#     else:
#         assert False, "No such parameter: {}".format(param_name)
#         return None

# def gen_bytes_str(tensor):
#     bytes = tensor.numel() * tensor.element_size()
#     return "[DIST BYTES]:  {} bytes".format(bytes) 

# class DistInfoGenerator(object):
#     @staticmethod
#     def gen_broadcast_(args, kwargs, info_type: INFOTYPE):
#         if info_type == INFOTYPE.BEFORE:
#             return "[DIST START_SYMBOL]: torch.distributed.broadcast"
#         elif info_type == INFOTYPE.AFTER:
#             return "[DIST END_SYMBOL]: torch.distributed.broadcast"
#         elif info_type == INFOTYPE.DATA:
#             Logger.error("Not implemented yet")
#             return str("")
#             # tensor = get_param(args, kwargs, 0, "tensor")
#             # return gen_bytes_str(tensor)

#     @staticmethod
#     def gen_all_reduce_(args, kwargs, info_type: INFOTYPE):
#         if info_type == INFOTYPE.BEFORE:
#             return "[DIST START_SYMBOL]: torch.distributed.all_reduce"
#         elif info_type == INFOTYPE.AFTER:
#             return "[DIST END_SYMBOL]: torch.distributed.all_reduce"
#         elif info_type == INFOTYPE.DATA:
#             Logger.error("Not implemented yet")
#             return str("")
#             # tensor = get_param(args, kwargs, 0, "tensor")
#             # return gen_bytes_str(tensor)
#         return str("")

#     @staticmethod
#     def gen_barrier(args, kwargs, info_type: INFOTYPE):
#         if info_type == INFOTYPE.BEFORE:
#             return "[DIST START_SYMBOL]: torch.distributed.barrier"
#         elif info_type == INFOTYPE.AFTER:
#             return "[DIST END_SYMBOL]: torch.distributed.barrier"
#         elif info_type == INFOTYPE.DATA:
#             Logger.error("Not implemented yet")
#             return str("")
#         return str("")

#     @staticmethod
#     def gen_allgather_(args, kwargs, info_type: INFOTYPE):
#         if info_type == INFOTYPE.BEFORE:
#             return "[DIST START_SYMBOL]: torch.distributed.all_gather"
#         elif info_type == INFOTYPE.AFTER:
#             return "[DIST END_SYMBOL]: torch.distributed.all_gather"
#         elif info_type == INFOTYPE.DATA:
#             Logger.error("Not implemented yet")
#             return str("")
#         return str("")

#     @staticmethod
#     def gen__allgather_base_(args, kwargs, info_type: INFOTYPE):
#         if info_type == INFOTYPE.BEFORE:
#             return "[DIST START_SYMBOL]: torch.distributed._all_gather_base"
#         elif info_type == INFOTYPE.AFTER:
#             return "[DIST END_SYMBOL]: torch.distributed._all_gather_base"
#         elif info_type == INFOTYPE.DATA:
#             Logger.error("Not implemented yet")
#             return str("")
#         return str("")

#     @staticmethod
#     def gen__reduce_scatter_base_(args, kwargs, info_type: INFOTYPE):
#         if info_type == INFOTYPE.BEFORE:
#             return "[DIST START_SYMBOL]: torch.distributed._reduce_scatter_base"
#         elif info_type == INFOTYPE.AFTER:
#             return "[DIST END_SYMBOL]: torch.distributed._reduce_scatter_base"
#         elif info_type == INFOTYPE.DATA:
#             Logger.error("Not implemented yet")
#             return str("")
#         return str("")


#     @staticmethod
#     def gen_send(args, kwargs, info_type: INFOTYPE):
#         if info_type == INFOTYPE.BEFORE:
#             return "[DIST START_SYMBOL]: torch.distributed.send"
#         elif info_type == INFOTYPE.AFTER:
#             return "[DIST END_SYMBOL]: torch.distributed.send"
#         elif info_type == INFOTYPE.DATA:
#             Logger.error("Not implemented yet")
#             return str("")
#         return str("")

#     @staticmethod
#     def gen_recv_(args, kwargs, info_type: INFOTYPE):
#         if info_type == INFOTYPE.BEFORE:
#             return "[DIST START_SYMBOL]: torch.distributed.recv"
#         elif info_type == INFOTYPE.AFTER:
#             return "[DIST END_SYMBOL]: torch.distributed.recv"
#         elif info_type == INFOTYPE.DATA:
#             Logger.error("Not implemented yet")
#             return str("")
#         return str("")

# # op_name: all_reduce_
# def gen_dist_op_str(op_name, args, kwargs, info_type: INFOTYPE):
#     gen_func_name = "gen_{}".format(op_name)
#     if hasattr(DistInfoGenerator, gen_func_name):
#         func = getattr(DistInfoGenerator, gen_func_name)
#         return func(args, kwargs, info_type)
#     else:
#         Logger.error("No such function: {}".format(gen_func_name))
#         # assert False, "No such function: {}".format(gen_func_name)
#     return str("")

# class DelimiterGenerator(object):
#     @staticmethod
#     def gen_delimiter_before_operation(name, args, kwargs):
#         if name.startswith('c10d'):
#             dist_op_name = get_dist_op_name(name)
#             return gen_dist_op_str(dist_op_name, args, kwargs, INFOTYPE.BEFORE)
#         else:
#             return "[START_SYMBOL]: {}".format(name)

#     @staticmethod
#     def gen_delimiter_after_operation(name, args, kwargs):
#         if name.startswith('c10d'):
#             dist_op_name = get_dist_op_name(name)
#             return gen_dist_op_str(dist_op_name, args, kwargs, INFOTYPE.AFTER)
#         else:
#             return "[END_SYMBOL]: {}".format(name)
        