cmake_minimum_required(VERSION 3.12)
project(gemm_benchmark)

# Find CUDA package
find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARY NAMES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Add CUTLASS include directory
include_directories(${CMAKE_SOURCE_DIR}/cutlass/include)  

# Set the CUDA architecture (Ampere - compute capability 8.6)
set(CUDA_NVCC_ARCH_FLAGS "-gencode=arch=compute_86,code=sm_86")

# Set CUDA flags and other compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_NVCC_ARCH_FLAGS} -lcublas")
include_directories(${CUBLAS_INCLUDE_DIRS})


# Include directories for header files
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUBLAS_INCLUDE_DIRS})

# Add executable and specify source files
cuda_add_executable(gemm
        ${CMAKE_SOURCE_DIR}/gemm_benchmark.cu
        ${CMAKE_SOURCE_DIR}/src/matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/my_gemm.cu
        ${CMAKE_SOURCE_DIR}/src/kernels_calls.cu
        ${CMAKE_SOURCE_DIR}/src/wrapper.cu
)

# Set CUDA as the language for the main CUDA file
set_property(TARGET gemm PROPERTY CUDA_STANDARD 12)

# Link cuBLAS library
target_link_libraries(gemm ${CUDA_CUBLAS_LIBRARIES})    

cuda_add_executable(batched_gemm
        ${CMAKE_SOURCE_DIR}/batched_gemm_benchmark.cu
        ${CMAKE_SOURCE_DIR}/src/matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/my_gemm.cu
        ${CMAKE_SOURCE_DIR}/src/kernels_calls.cu
        ${CMAKE_SOURCE_DIR}/src/wrapper.cu
)

# Set CUDA as the language for the main CUDA file
set_property(TARGET batched_gemm PROPERTY CUDA_STANDARD 12)

# Link cuBLAS library
target_link_libraries(batched_gemm ${CUDA_CUBLAS_LIBRARIES} ${CUTLASS_LIB})   

find_program(CLANG_FORMAT_BIN clang-format)

if(CLANG_FORMAT_BIN)
    file(GLOB_RECURSE PROJECT_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.cu
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/*.cu
    )

    add_custom_target(format
        COMMAND ${CLANG_FORMAT_BIN}
        -i
        ${PROJECT_SOURCE_FILES}
    )
endif()
