file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_executable(Benchmark ${CPP_SOURCES} ${CU_SOURCES})

set_target_properties(Test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON   # Builds device code in multiple passes (useful for device linking).
    CUDA_ARCHITECTURES "75"         # Specify architecture; 75 = Turing, 80 = Ampere, etc.
)

find_package(benchmark CONFIG REQUIRED)
target_link_libraries(Benchmark PRIVATE benchmark::benchmark benchmark::benchmark_main)
#target_compile_options(Benchmark PRIVATE /wd1394)