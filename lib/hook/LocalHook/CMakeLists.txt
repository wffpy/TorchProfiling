if ((CUDA_DEV STREQUAL TRUE) AND CUDA_FOUND)
    add_library(LocalHook STATIC LocalHook.cpp)
endif()