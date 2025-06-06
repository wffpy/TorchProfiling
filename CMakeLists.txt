cmake_minimum_required(VERSION 3.5...3.26)
project (TorchProfiling)

cmake_policy(SET CMP0075 NEW)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_compile_options(-w -std=c++14 -O2)

find_package(CUDA)
message("CUDA_DEV: ${CUDA_DEV}")
if ((CUDA_DEV STREQUAL TRUE) AND CUDA_FOUND)
    ADD_DEFINITIONS(-DCUDA_DEV)
    message(STATUS "CUDA device is enabled")
    message(STATUS "CUDA INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA LINK_DIRS: ${CUDA_LINK_DIRS}")
    include_directories(${CUDA_INCLUDE_DIRS})
    message("CUDA_LIBS: ${CUDA_LIBRARIES}")
else ()
    message(STATUS "CUDA is not enabled")
endif()

if (USE_CAPSTONE STREQUAL TRUE)
    ADD_DEFINITIONS(-DUSE_CAPSTONE)
    message("CAPSTONE is enabled!!!!!!!!!!")
endif()

if (XPU_DEV)
    message("XPU_DEB: ${XPU_DEV}")
    ADD_DEFINITIONS(-DXPU_DEV)
    find_library(XPU_LIB xpurt PATHS /usr/local/xcudart/lib)
    message("XPU_LIB: ${XPU_LIB}")
endif()

find_package(Python3 REQUIRED Interpreter COMPONENTS Development)
find_package(PythonInterp REQUIRED)

if (Python3_FOUND)
    message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Python3 not found")
endif()

set(PYTHON_EXECUTABLE python3)

#cmake_path(SET python3_path NORMALIZE "${Python3_INCLUDE_DIRS}/../..")
set(python3_path "${Python3_INCLUDE_DIRS}/../..")
message("python3_path: ${python3_path}")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${python3_path})
set(pybind11_DIR ${PYBIND11_GET_OPINFO}/share/cmake/pybind11)
message("pybind11_DIR: ${pybind11_DIR}")

find_package(pybind11 CONFIG REQUIRED)

include_directories(
    ${pybind11_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)

#set (CMAKE_INSTALL_PREFIX "/ssd1/wangfangfei/projects/install")
#set(DST_DIRECTORY_TO_INSTALL ${CMAKE_INSTALL_PREFIX}/TorchProfiling)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${DST_DIRECTORY_TO_INSTALL})
