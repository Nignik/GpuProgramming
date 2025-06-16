file(GLOB_RECURSE CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_executable(Test ${CPP_SOURCES} ${CU_SOURCES})

set_target_properties(Test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

enable_testing()

find_package(GTest CONFIG REQUIRED)
target_link_libraries(Test PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

add_test(Test Test)