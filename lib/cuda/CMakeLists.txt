if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_library(GpuHook STATIC
        GpuProfiler.cpp
    )
endif()

