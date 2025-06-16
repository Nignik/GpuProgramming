cmake_minimum_required(VERSION 3.20)

project(GpuProgramming LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmark)
