#ifndef KERNELS_CALLS_H
#define KERNELS_CALLS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "my_gemm.h"

// cublas matrix multiplication kernel
template <typename T>
T* call_cublasGemm(const T* h_A, const T* h_B, int N);

// Matrix multiplication kernel
template <typename T>
T* call_GemmGlobalMemory(const T* h_A, const T* h_b, int N);

// Matrix multiplication kernel with shared memory
template <typename T>
T* call_GemmSharedMemory(const T* h_A, const T* h_b, int N);

// cublas Batched matrix multiplication kernel
template <typename T>
T* call_cublasGemmBatched(const T* h_A, const T* h_B, int N, int BATCH_SIZE);

// Batched gemm kernel
template <typename T>
T* call_BatchedGemmGlobalMemory(const T* h_A, const T* h_B, int N,
                                int BATCH_SIZE);

// Batched gemm kernel with shared memory
template <typename T>
T* call_BatchedGemmSharedMemory(const T* h_A, const T* h_B, int N,
                                int BATCH_SIZE);

// cutlass Batched matrix multiplication kernel
template <typename T>
T* call_CutlassBatchedGemm(const T* h_A, const T* h_B, int N, int BATCH_SIZE);

#endif  // CODE_H
