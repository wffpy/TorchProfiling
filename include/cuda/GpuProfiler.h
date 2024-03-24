#ifndef GPUPROFILER_H
#define GPUPROFILER_H
#include <iostream>

namespace gpu_profiler {

class GpuHookWrapper {
public:
    GpuHookWrapper() {}
    ~GpuHookWrapper() {}

    static int local_cuda_launch(void* func);

    int (*oriign_cuda_launch_)(void*) = nullptr;
};

}   // namespace gpu_profiler

#endif