import module_logging as ml
import torch


def test_trace():
    # Set up two tensors
    tensor1 = torch.rand(3, 3)  # Random tensor of shape 3x3
    tensor2 = torch.rand(3, 3)  # Another random tensor of shape 3x3

    # Move tensors to GPU if available
    with ml.trace.Tracer(path="./profiling1.log", print_module_info=False, ranks=[0, 1, 2]):
        result = tensor1 + tensor2
