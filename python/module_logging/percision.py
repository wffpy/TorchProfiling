import h5py
import torch
import os
from logging import Logger

class PercisionDebugger:
    """
    insert delimiters before and and after op execution
    """

    def __init__(self) -> None:
        self.path = "/tmp/data.hf5"
        self.step = 0
        self.record_steps = None
        self.model = None
        self.hook_handles = []
        self.forward_input_hook_handles = []
        self.forward_output_hook_handles = []
        self.backward_input_hook_handles = []
        self.backward_output_hook_handles = []
        self.weight_hook_handles = []
        self.grad_hook_handles = []
        self.h5f = None
        self.ranks = None
        self.rank = None
        self.name_dict = {}
        self.saved = []
    
    def config(self, model,  path="/tmp/", steps=None, ranks=[]):
        self.model = model
        self.ranks = ranks
        self.rank = int(os.environ["RANK"])
        if path is not None and self.rank in self.ranks:
            self.path = path + "rank_" + str(rank) + ".hf5"
    
        if steps is not None:
            self.record_steps = steps
    
    def is_active(self):
        return self.rank in self.ranks and self.step in self.record_steps

    def pre_forward_hook_wrapper(self, module_name):
        def pre_forward_hook(module, input):
            if not self.is_active():
                return

            index =0
            for t in input:
                if not isinstance(t, torch.Tensor):
                    continue
                input_name = "step_" + str(self.step) + "_" + module_name + "_forward_input_" + str(index)
                if input_name in self.saved:
                    Logger.error("duplicate key: {}".format(input_name))
                Logger.info("save module input: {}".format(input_name))
                self.h5f.create_dataset(input_name, data=t.cpu().detach().numpy())
                self.saved.append(input_name)
                index += 1

            for name, param in module.named_parameters():
                param_name = "step_" + str(self.step) + "_" + module_name + "_forward_param_" + name
                if param_name in self.saved:
                    Logger.error("duplicate key: {}".format(param_name))
                if not isinstance(param, torch.Tensor):
                    continue
                Logger.info("save module param: {}".format(param_name))
                self.h5f.create_dataset(param_name, data=param.cpu().detach().numpy())
                self.saved.append(param_name)

        return pre_forward_hook

    def post_forward_hook_wrapper(self, module_name):
        def post_forward_hook(module, input, output):
            if not self.is_active():
                return
            index = 0
            if isinstance(output, tuple):
                for t in output:
                    output_name = "step_" + str(self.step) + "_" +  module_name + "_forward_output_" + str(index)
                    if output_name in self.saved:
                        Logger.error("duplicate key: {}".format(output_name))
                    if not isinstance(t, torch.Tensor):
                        continue
                    Logger.info("save module output: {}".format(output_name))
                    self.h5f.create_dataset(output_name, data=t.cpu().detach().numpy())
                    self.saved.append(output_name)
                    index += 1
            elif isinstance(output, torch.Tensor):
                output_name = "step_" + str(self.step) + module_name + "forward_output_" + str(index)
                if output_name in self.saved:
                    Logger.error("duplicate key: {}".format(output_name))
                Logger.info("save module output: {}".format(output_name))
                self.h5f.create_dataset(output_name, data=output.cpu().detach().numpy())
                self.saved.append(output_name)

        return post_forward_hook

    def pre_backward_hook_wrapper(self, module_name):
        def pre_backward_hook(module, input):
            if not self.is_active():
                return
            index = 0
            if isinstance(input, tuple):
                for t in input:
                    input_name = "step_" + str(self.step) + "_" + module_name + "_backward_input_" + str(index)
                    if input_name in self.saved:
                        Logger.error("duplicate key: {}".format(input_name))
                    if t is None or not isinstance(t, torch.Tensor):
                        continue
                    Logger.info("save module input: {}".format(input_name))
                    self.h5f.create_dataset(input_name, data=t.cpu().detach().numpy())
                    self.saved.append(input_name)
                    index += 1
            elif isinstance(input, torch.Tensor):
                input_name = "step_" + str(self.step) + module_name + "backward_input_" + str(index)
                if input_name in self.saved:
                    Logger.error("duplicate key: {}".format(input_name))
                Logger.info("save module input: {}".format(input_name))
                self.h5f.create_dataset(input_name, data=input.cpu().detach().numpy())
                self.saved.append(input_name)

        return pre_backward_hook

    def post_backward_hook_wrapper(self, module_name):
        def post_backward_hook(module, input, output):
            if not self.is_active():
                return
            index = 0
            if isinstance(output, tuple):
                for t in output:
                    output_name = "step_" + str(self.step) + "_" + module_name + "_backward_output_" + str(index)
                    if output_name in self.saved:
                        Logger.error("duplicate key: {}".format(output_name))
                    if t is None or not isinstance(t, torch.Tensor):
                        continue
                    Logger.info("save module output: {}".format(output_name))
                    self.h5f.create_dataset(output_name, data=t.cpu().detach().numpy())
                    self.saved.append(output_name)
                    index += 1
            elif isinstance(output, torch.Tensor):
                output_name = "step_" + str(self.step) + "_" + module_name + "backward_output_" + str(index)
                if output_name in self.saved:
                    Logger.error("duplicate key: {}".format(output_name))
                Logger.info("save module output: {}".format(output_name))
                self.h5f.create_dataset(output_name, data=output.cpu().detach().numpy())
                self.saved.append(output_name)

        return post_backward_hook

    def save_forward_input(self, module_name=None):
        for name, module in self.model.named_modules():
            if module_name is None:
                handle = module.register_forward_pre_hook(self.pre_forward_hook_wrapper(name))
                self.forward_input_hook_handles.append(handle)
            elif module_name in name:
                handle = module.register_forward_pre_hook(self.pre_forward_hook_wrapper(name))
                self.forward_input_hook_handles.append(handle)
            
    def remove_forward_input(self):
        for handle in self.forward_input_hook_handles:
            handle.remove()
    
    def save_forward_output(self, module_name=None):
        for name, module in self.model.named_modules():
            if module_name is None:
                handle = module.register_forward_hook(self.post_forward_hook_wrapper(name))
                self.forward_output_hook_handles.append(handle)
            elif module_name in name:
                handle = module.register_forward_hook(self.post_forward_hook_wrapper(name))
                self.forward_output_hook_handles.append(handle)

    def remove_forward_output(self):
        for handle in self.forward_output_hook_handles:
            handle.remove()

    def save_backward_input(self, module_name=None):
        for name, module in self.model.named_modules():
            if module_name is None:
                handle = module.register_full_backward_pre_hook(self.pre_backward_hook_wrapper(name))
                self.backward_input_hook_handles.append(handle)
            elif module_name in name:
                handle = module.register_full_backward_pre_hook(self.pre_backward_hook_wrapper(name))
                self.backward_input_hook_handles.append(handle)
    
    def remove_backward_input(self):
        for handle in self.backward_input_hook_handles:
            handle.remove()
    
    def save_backward_output(self, module_name=None):
        for name, module in self.model.named_modules():
            if module_name is None:
                handle = module.register_full_backward_hook(self.post_backward_hook_wrapper(name))
                self.backward_output_hook_handles.append(handle)
            elif module_name in name:
                handle = module.register_full_backward_hook(self.post_backward_hook_wrapper(name))
                self.backward_output_hook_handles.append(handle)
    
    def remove_backward_output(self):
        for handle in self.backward_output_hook_handles:
            handle.remove()
    
    def save_weights(self, after_optimizer=False):
        if not self.is_active():
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_name = "step_" + str(self.step) + "_weight_" + name
                if after_optimizer:
                    param_name = "step_" + str(self.step) + "_weight_updated_" + name
                if param_name in self.saved:
                    Logger.error("duplicate key: {}".format(param_name))
                if param.grad is None or not isinstance(param, torch.Tensor):
                    continue
                Logger.info("save weight: {}".format(param_name))
                self.h5f.create_dataset(param_name, data=param.cpu().detach().numpy())
                self.saved.append(param_name)
    
    def save_grads(self):
        if not self.is_active():
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_name = "step_" + str(self.step) + "_weight_grad_" + name
                if param_name in self.saved:
                    Logger.error("duplicate key: {}".format(param_name))
                if not isinstance(param.grad, torch.Tensor):
                    continue
                Logger.info("save module grad: {}".format(param_name))
                self.h5f.create_dataset(param_name, data=param.grad.cpu().detach().numpy())
                self.saved.append(param_name)

    def save_tensor(self, tensor, name):
        if not self.is_active():
            return
        tensor_name = "step_" + str(self.step) + "_" + name
        Logger.info("save tensor: {}".format(tensor_name))
        self.h5f.create_dataset(name, data=tensor.cpu().detach().numpy())
        self.saved.append(tensor_name)
    
    # this function should be called after optimzer.step()
    def update_step(self):
        self.step += 1

    def __enter__(self):
        if self.is_not_active_rank():
            return
        if self.h5f is None:
            self.h5f = h5py.File(self.path, 'w')
            Logger.info('open file: {}'.format(self.path))
        
        self.save_forward_input()
        self.save_forward_output()
        self.save_backward_input()
        self.save_backward_output()
        Logger.info("persion debugger enter")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_not_active_rank():
            return
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None
        Logger.info('close file: {}'.format(self.path))

        if len(self.forward_input_hook_handles) > 0:
            self.remove_forward_input()
        if len(self.forward_output_hook_handles) > 0:
            self.remove_forward_output()
        if len(self.backward_input_hook_handles) > 0:
            self.remove_backward_input()
        if len(self.backward_output_hook_handles) > 0:
            self.remove_backward_output()
        Logger.info('remove hooks')

        Logger.info("persion debugger exit")
    def is_not_active_rank(self):
        return self.rank not in self.ranks
        

debugger = PercisionDebugger()