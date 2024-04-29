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
# import pandas as pd

# class OPTYPE(Enum):
#     ALLREDUCE = auto()
#     BARRIER = auto()
#     BROADCAST = auto()
#     ALLGATHER = auto()
#     REDUCESCATTER = auto()
    

# class BkclDataReader(object):
#     def __init__(self, excel_file_path):
#         self.xls = pd.ExcelFile(excel_file_path)
#         self.sheet_names = self.xls.sheet_names
#         self.sheet_count = len(self.sheet_names)

#     def read(op_name:str, data_type:str):
#         '''
#         read the data from excel file
#         '''
#         sheet_name = ""
#         for name in self.sheet_names:
#             if op_name in name and data_type in name:
#                 sheet_name = name
#                 break
        
#         df = self.xls.readk_excel(xls, sheet_name)
#         bytes = df[""]
    

# class OpPerfEstimate(object):
#     '''
#     the base class of all the dist op analysis
#     '''
#     def __init__(self):
#         pass
    
#     def __call__(self, op_name):
#         '''
#         the main function to estimate the time cost of a specific operator
#         '''
#         return self.estimate()

#     def estimate():
#         '''
#         estimate the performance of a specific operator
#         '''
#         pass


# class AllReduce(OpPerfEstimate):  
#     def __init__(self, tensor: torch.Tensor):
#         self.tensor = tensor
#         self.bytes_range =[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456]
#         self.bw = [0.09, 0.18, 0.32, 0.63, 1.25, 2.43, 4.61, 8.50, 14.91, 23.06, 31.59, 38.46, 39.42, 40.10, 40.51, 41.66, 42.16, 42.55, 42.06]
    
#     def estimate(self):
#         '''
#         estimate the time cost of all reduce op
#         '''
#         bytes = self.tensor.numel() * self.tensor.element_size()
#         assert(len(self.bytes_range) == len(self.bw), "the length of bytes_range and bw should be the same")
        
#         dtype = self.tensor.dtype 
#         if isinstance(dtype, torch.float):
#             pass
#         elif isinstance(dtype, torch.float16):
#             pass
#         elif isinstance(dtype, torch.bfloat16):
#             pass

origin_all_reduce = None
origin_broadcast = None
origin_barrier = None
origin__all_gather_base = None
origin__reduce_scatter_base = None
origin_all_gather = None
origin_send = None
origin_recv = None
        
def mock_all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    print("[DIST START_SYMBOL]: torch.distributed.all_reduce", flush=True)
    bytest = tensor.numel() * tensor.element_size()
    print("[DIST BYTES]: {} bytes".format(bytest), flush=True)
    ret = origin_all_reduce(tensor, op, group, async_op)
    print("[DIST END_SYMBOL]: torch.distributed.all_reduce", flush=True)
    return ret

def mock_broadcast(tensor, src, group=None, async_op=False):
    print("[DIST START_SYMBOL]: torch.distributed.broadcast", flush=True)
    bytest = tensor.numel() * tensor.element_size()
    print("[DIST BYTES]:{} bytes".format(bytest), flush=True)
    ret = origin_broadcast(tensor, src, group, async_op)
    print("[DIST END_SYMBOL]: torch.distributed.broadcast", flush=True)
    return ret

def mock_barrier(group=None, async_op=False, device_ids=None):
    print("[DIST START_SYMBOL]: torch.distributed.barrier", flush=True)
    ret = origin_barrier(group, async_op, device_ids)
    print("[DIST END_SYMBOL]: torch.distributed.barrier", flush=True)
    return ret

# def mock_all_gather(tensor_list, tensor, group=None, async_op=False):
#     return
def mock_all_gather(tensor_list, tensor, group=None, async_op=False):
    print("[START_SYMBOL]: torch.distributed.all_gather", flush=True)
    bytest = tensor.numel() * tensor.element_size()
    print("[DIST BYTES]:{} bytes".format(bytest), flush=True)
    ret = origin_all_gather(tensor_list, tensor, group, async_op)
    print("[END_SYMBOL]: torch.distributed.all_gather", flush=True)
    return ret

def mock__all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
    print("[START_SYMBOL]: torch.distributed._all_gather_base", flush=True)
    bytest = output_tensor.numel() * output_tensor.element_size()
    print("[DIST BYTES]:{} bytes".format(bytest), flush=True)
    ret = origin__all_gather_base(output_tensor, input_tensor, group, async_op)
    print("[END_SYMBOL]: torch.distributed._all_gather_base", flush=True)
    return ret

def mock__reduce_scatter_base(
    output_tensor, input, op=ReduceOp.SUM, group=None, async_op=False
):
    print("[DIST START_SYMBOL]: torch.distributed._reduce_scatter_base", flush=True)
    bytest = output_tensor.numel() * output_tensor.element_size()
    print("[DIST BYTES]:{} bytes".format(bytest), flush=True)
    ret = origin__reduce_scatter_base(output_tensor, input, op, group, async_op)
    print("[DIST END_SYMBOL]: torch.distributed._reduce_scatter_base", flush=True)
    return ret

def mock_send(tensor: torch.Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: int = 0) -> None:
    print("[DIST START_SYMBOL]: torch.distributed.send", flush=True)
    bytest = tensor.numel() * tensor.element_size()
    print("[DIST BYTES]:{} bytes".format(bytest), flush=True)
    ret = origin_send(tensor, dst, group, tag)
    print("[DIST END_SYMBOL]: torch.distributed.send", flush=True)
    return ret 

def mock_recv(tensor: torch.Tensor, src: Optional[int] = None, group: Optional[ProcessGroup] = None, tag: int = 0) -> None:
    print("[DIST START_SYMBOL]: torch.distributed.recv", flush=True)
    bytest = tensor.numel() * tensor.element_size()
    print("[DIST BYTES]:{} bytes".format(bytest), flush=True)
    ret = origin_recv(tensor, src, group, tag)
    print("[DIST END_SYMBOL]: torch.distributed.recv", flush=True)
    return  ret

def monkey_patch():
    '''
    Dist Op Monkey Patch
    '''
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
    
    # we should apply monkey patch immediately after we save original function
    
    torch.distributed.all_reduce = mock_all_reduce
    torch.distributed.broadcast = mock_broadcast
    torch.distributed.barrier = mock_barrier
    torch.distributed._all_gather_base = mock__all_gather_base
    torch.distributed._reduce_scatter_base = mock__reduce_scatter_base
    torch.distributed.all_gather = mock_all_gather
    torch.distributed.send = mock_send
    torch.distributed.recv = mock_recv


