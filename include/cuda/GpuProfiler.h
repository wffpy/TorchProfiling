#ifndef GPUPROFILER_H
#define GPUPROFILER_H
#include <iostream>
// #include "Utils.h" 
// #include <cuda.h>
// #include <cuda_runtime.h>

namespace gpu_profiler {
void cupti_activity_init();

void cupti_activity_flush();

void cupti_activity_finalize();

void register_gpu_hook();

}   // namespace gpu_profiler

#endif
