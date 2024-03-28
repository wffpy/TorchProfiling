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

### 1. Get Profiling Data
#### step 1: Profiling

##### Usage 1: not display torch.Module
```
import module_logging as m
m.Hook.install_hook()

with m.logger.PerformanceLogger():
    model()

```

##### Usage 2: display the torch.Module
```
import module_logging as m
m.Hook.install_hook()

m = model
with m.logger.PerformanceLogger(m):
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

```

### 2. 统计C函数调用次数
```
import module_logging
module_logging.Hook.install_hook()

python test.py
```

### C Function Counter
TODO:
use a different hook function which will try to modify the hooked function's assembly code.
