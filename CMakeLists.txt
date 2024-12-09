cmake_minimum_required(VERSION 3.25)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)

project(hello-cuda LANGUAGES CUDA CXX C)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

set(HELLO_CUDA_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(HELLO_CUDA_ROOT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(cmake/CPM.cmake)
include(cmake/Targets.cmake)

add_subdirectory(src)

enable_testing()
add_subdirectory(test)
