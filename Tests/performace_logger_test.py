import torch
import module_logging as ml

def test_trace():
    # Set up two tensors
    tensor1 = torch.rand(3, 3)
    tensor2 = torch.rand(3, 3)

    with ml.PerformanceLogger():
        result = tensor1 + tensor2

 