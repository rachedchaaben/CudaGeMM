#ifndef KERNELS_CALLS_H
#define KERNELS_CALLS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "my_gemm.h"

// cublas matrix multiplication kernel
template <typename T>
T* call_cublasGemm(const T* h_A, const T* h_B, int N);

// Matrix multiplication kernel
double* call_DgemmGlobalMemory(const double* h_A, const double* h_b, int N);

// Matrix multiplication kernel
float* call_SgemmGlobalMemory(const float* h_A, const float* h_b, int N);

// Matrix multiplication kernel with shared memory
double* call_DgemmSharedMemory(const double* h_A, const double* h_b, int N);

// Matrix multiplication kernel with shared memory
float* call_SgemmSharedMemory(const float* h_A, const float* h_b, int N);

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
