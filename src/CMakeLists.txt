file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_executable(GpuProgramming ${CPP_SOURCES} ${CU_SOURCES})

set_target_properties(GpuProgramming PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON   # Builds device code in multiple passes (useful for device linking).
    CUDA_ARCHITECTURES "75"         # Specify architecture; 75 = Turing, 80 = Ampere, etc.
)
