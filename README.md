# Performance Logger

## Build And Install
### For Cpu
```
// build with cpp extension
// try to hook some function
bash scripts/build.sh

// build without cpp extension
// just profiling cpu kernel, and analysis the log
bash scripts/build_regular.sh
```

### For Gpu
```
export CUDA_DEV=true
bash scripts/build.sh
```

## User Guide

### 1. Get Profiling Data
#### step 1: Profiling

##### Usage 1: not display torch.Module
```
import module_logging as ml

with ml.logger.combined_context():
    model()

```

##### Usage 2: display the torch.Module
```
import module_logging as ml

m = model
with ml.logger.combined_context(m):
    m()

```

#### step 2: Post-Processing
```
# for default print the total time table
python -m module_logging --path 7.log

# print summary table
python -m module_logging --path 7.log --summary

# print the detail table
python -m module_logging --path 7.log --detail

# print all 3 kinds table
python -m module_logging --path 7.log --all

# write table to csv: /tmp/total.csv
python -m module_logging --path 7.log --csv

#compare mode
python -m module_logging --compare --lhs_path 0.log --rhs_path 1.log

# compare mode and write to csv: /tmp/compare.csv
python -m module_logging --compare --lhs_path 0.log --rhs_path 1.log --csv

```

### 2. 统计C函数调用次数
```
import module_logging
module_logging.Hook.install_hook()

python test.py
```
![image](https://github.com/wffpy/TorchProfiling/blob/main/IMG/count.jpg)

### 3. 打印C函数的调用栈
```
export PRINT_BACKTRACE=true

import module_logging
module_logging.Hook.install_hook()

python test.py
```
example:
```
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/module_logging/Hook.cpython-38-x86_64-linux-gnu.so(_ZN5trace6Tracer5traceEv+0x39) [0x7fb56afa46d9]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/module_logging/Hook.cpython-38-x86_64-linux-gnu.so(_ZN5trace6TracerC1ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x92) [0x7fb56afa4942]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/module_logging/Hook.cpython-38-x86_64-linux-gnu.so(_ZN14CpuHookWrapper20local_launch_arg_setEPKvmm+0x99) [0x7fb56afa2b69]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch_xmlir/libxdnn_pytorch.so(_ZN14xpukernel_xpu310calc_basicILi2EfEEvPKT0_S3_PS1_x+0x46) [0x7fb69f724076]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch_xmlir/libxdnn_pytorch.so(+0x3c44692) [0x7fb6a23d4692]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch_xmlir/libxdnn_pytorch.so(_ZN8xpytorch3xpu3api13broadcast_mulIfEEiPNS1_7ContextEPKT_S7_PS5_RKSt6vectorIlSaIlEESD_+0x4b) [0x7fb6a23d26db]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch_xmlir/libxdnn_pytorch.so(+0x1a139ca) [0x7fb6a01a39ca]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch_xmlir/libxdnn_pytorch.so(_ZN12xdnn_pytorch10mul_tensorEPN8xpytorch3xpu3api7ContextERKNS_6TensorES7_RS5_+0x1f5) [0x7fb6a01a0685]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch_xmlir/_XMLIRC.cpython-38-x86_64-linux-gnu.so(+0xc5a1d4) [0x7fb6ed9761d4]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch_xmlir/_XMLIRC.cpython-38-x86_64-linux-gnu.so(+0xe4ae6e) [0x7fb6edb66e6e]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so(_ZN2at4_ops10mul_Tensor10redispatchEN3c1014DispatchKeySetERKNS_6TensorES6_+0x8a) [0x7fb7ce23204a]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so(+0x3d09390) [0x7fb7cffeb390]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so(+0x3d09e9b) [0x7fb7cffebe9b]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so(_ZN2at4_ops10mul_Tensor4callERKNS_6TensorES4_+0x175) [0x7fb7ce29b715]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so(+0x526184b) [0x7fb7d154384b]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so(_ZN5torch8autograd9generated12PowBackward05applyEOSt6vectorIN2at6TensorESaIS5_EE+0x144) [0x7fb7cfee50c4]
/root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so(+0x48d9d8b) [0x7fb7d0bbbd8b]
```


### C Function Counter
TODO:
use a different hook function which will try to modify the hooked function's assembly code.
