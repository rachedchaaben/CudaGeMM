#ifndef KERNELS_CALLS_H
#define KERNELS_CALLS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels.h"

// cublas matrix multiplication kernel template
template <typename T>
T* call_cublasGemm(const T* h_A, const T* h_B, int N);

// matrix multiplication kernel double precision: naive implementation
double* call_DgemmGlobalMemory(const double* h_A, const double* h_b, int N);

// matrix multiplication kernel single precision: naive implementation
float* call_SgemmGlobalMemory(const float* h_A, const float* h_b, int N);

// matrix multiplication kernel double precision: using shared memory
double* call_DgemmSharedMemory(const double* h_A, const double* h_b, int N);

// matrix multiplication kernel double precision: using shared memory and 2D
// threads Blocktiling
float* call_SgemmSharedMemory(const float* h_A, const float* h_b, int N);

// cutlass matrix multiplication kernel single precision
float* call_cutlassSgemm(const float* h_A, const float* h_B, int N);

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
