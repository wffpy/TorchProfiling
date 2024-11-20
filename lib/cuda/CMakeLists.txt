if ((CUDA_DEV STREQUAL TRUE) AND CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

add_library(GpuHook STATIC GpuProfiler.cpp)
