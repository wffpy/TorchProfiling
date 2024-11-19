import torch


def test_tensor_tracer():
    tensor1 = torch.tensor([1, 2, 3], device='cpu').float()
    tensor2 = torch.tensor([4, 5, 6], device='cpu').float()

    from  module_logging import tensor_tracer 

    tensor_tracer.__enter__()
    print(torch.max(tensor1).item())

    tensor_tracer.trace("tensor1", tensor1)


    tensor1.add_(tensor2)

    print("After in-place addition on CUDA:", tensor1)
    tensor_tracer.__exit__()
