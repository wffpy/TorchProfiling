#ifndef GPUPROFILER_H
#define GPUPROFILER_H
#include <iostream>
// #include "Utils.h" 
// #include <cuda.h>
#include <cuda_runtime.h>

namespace gpu_profiler {

// class GpuHookWrapper {
// public:
//     GpuHookWrapper() {}
//     ~GpuHookWrapper() {}

//     static int local_cuda_launch(void* func);
//     static int local_cuda_launch_kernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);

//     int (*oriign_cuda_launch_)(void*) = nullptr;
//     int (*oriign_cuda_launch_kernel_)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = nullptr;
// };

// typedef utils::Singleton<GpuHookWrapper> SingletonGpuHookWrapper;

void register_gpu_hook();

}   // namespace gpu_profiler

#endif
