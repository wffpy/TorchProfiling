# Performance Logger

## Build And Install
### For Cpu
```
bash build.sh

```

### For Gpu
```
export CUDA_DEV=true
bash build.sh
```

## User Guide

### step 1: Profiling

#### Usage 1: not display torch.Module
```
import module_logging as m
m.Hook.install_hook()

with m.logger.PerformanceLogger():
    model()

```

#### Usage 2: display the torch.Module
```
import module_logging as m
m.Hook.install_hook()

m = model
with m.logger.PerformanceLogger(m):
    m()

```

### step 2: Post-Processing
```
python -m module_logging --gpu_log 7.log
```


### C Function Counter
TODO:
use a different hook function which will try to modify the hooked function's assembly code.
