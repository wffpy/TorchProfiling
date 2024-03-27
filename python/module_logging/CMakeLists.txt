# add_library(Hook SHARED init.cpp)
# add_subdirectory(pybind11)
pybind11_add_module(Hook init.cpp)

target_include_directories(Hook PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(Hook PRIVATE hook)
target_link_libraries(Hook PRIVATE CpuHook)
target_link_libraries(Hook PRIVATE Util)
target_link_libraries(Hook PRIVATE GpuHook)


target_include_directories(Hook PUBLIC ${Python3_INCLUDE_DIRS})

target_link_libraries(Hook PRIVATE ${Python3_LIBRARIES})

if ((CUDA_DEV STREQUAL TRUE) AND CUDA_FOUND)
    target_link_libraries(Hook PRIVATE ${CUDA_LIBRARIES})
    set (CUDA_LINK_PATH /usr/local/cuda/lib64/) 
    target_link_directories(Hook PRIVATE ${CUDA_LINK_PATH})
    target_link_libraries(Hook PRIVATE cupti)
    target_link_libraries(Hook PRIVATE cudart)
    target_link_libraries(Hook PRIVATE cuda)
endif()


# install(TARGETS Hook DESTINATION ${DST_DIRECTORY_TO_INSTALL})

set_target_properties(Hook
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/module_logging
)
