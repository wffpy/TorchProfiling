import torch
from torch.utils._python_dispatch import TorchDispatchMode
import time

MODULE_COUNTER = 0

def get_module_index():
    global MODULE_COUNTER
    MODULE_COUNTER += 1
    return MODULE_COUNTER

class PerformanceLogger(TorchDispatchMode):
    """
    insert delimiters before and and after op execution
    """


    def __init__(self, gpu=False, model=None) -> None:
        super().__init__()
        self.gpu = gpu
        if model:
            if isinstance(model, list):
                for module in model:
                    for name, m in module.named_modules():
                        name += "_{}".format(get_module_index())
                        self._register_hook(name, m)
            elif isinstance(model, torch.nn.Module):
                for name, m in model.named_modules():
                    name += "_{}".format(get_module_index())
                    self._register_hook(name, m)

    def pre_forward_hook_wrapper(self, name):
        def pre_forward_hook(module, input):
            torch.cuda.synchronize()
            print("[BEGIN FORWARD]: {}".format(name))

        return pre_forward_hook

    def post_forward_hook_wrapper(self, name):
        def post_forward_hook(module, input, output):
            torch.cuda.synchronize()
            print("[END FORWARD]: {}".format(name))

        return post_forward_hook

    def pre_backward_hook_wrapper(self, name):
        def pre_backward_hook(module, input):
            torch.cuda.synchronize()
            print("[BEGINE BACKWARD]: {}_backward".format(name))

        return pre_backward_hook

    def post_backward_hook_wrapper(self, name):
        def post_backward_hook(module, input, output):
            torch.cuda.synchronize()
            print("[END BACKWARD]: {}_backward".format(name))

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
        print("[START_SYMBOL]: {}".format(str(op)))
        if self.gpu:
            # get device index and stream
            device_index = torch.cuda.current_device()
            device = torch.device("cuda:" + str(device_index))
            stream = torch.cuda.current_stream(device) 
            # end event
            event = torch.cuda.Event(enable_timing=True)
            # start event
            start_event = torch.cuda.Event(enable_timing=True)
            # insert start event on current stream
            stream.record_event(start_event)
            # call op
            output = op(*args, **kwargs)
            # insert end event on current stream
            stream.record_event(event)
            event.synchronize()
            duration = start_event.elapsed_time(event)
            print("[CUDA_PROF]: {}".format(duration))
        else:
            import os
            print("[PROCESS ID]: {}".format(os.getpid()))
            # call op
            output = op(*args, **kwargs)
            
        print("[END_SYMBOL]: {}".format(str(op)))
        return output

    # def __torch_dispatch__(self, op, types, args=(), kwargs=None):
    #     if kwargs is None:
    #         kwargs = {}
    #     #  insert pre-op delimiter
    #     print("[START_SYMBOL]: {}".format(str(op)))
    #     # start_time = time.time_ns()
    #     stream = torch.cuda.current_stream()
    #     event = torch.cuda.Event(enable_timing=True)
    #     start_event = torch.cuda.Event(enable_timing=True)
    #     stream.record_event(start_event)
    #     output = op(*args, **kwargs)
    #     stream.record_event(event)
    #     event.synchronize()
    #     duration = start_event.elapsed_time(event)
    #     # end_time = time.time_ns()
    #     # duration = end_time - start_time
    #     #  insert post-op delimiter
    #     print("DURATION: {}".format(duration))
    #     print("[END_SYMBOL]: {} ns".format(str(op)))
    #     return output


