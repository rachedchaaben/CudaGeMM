#ifndef MY_GEMM_H
#define MY_GEMM_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// reference CPU GeMM to check for correctness
float* gemm_cpu(float* A, float* B, int m, int n, int k);

// Matrix multiplication kernel using global memory
template <typename T>
__global__ void globalMemoryGemm(const T* A, const T* B, T* C, int N);

// Matrix multiplication kernel using shared memory
template <typename T>
__global__ void SharedMemoryGemm(const T* A, const T* B, T* C, int N);

// Batched matrix multiplication kernel using global memory
template <typename T>
__global__ void GlobalMemoryBatchedGemm(const T* A, const T* B, T* C, int N,
                                        int batch_size);

// Batched matrix multiplication kernel using shared memory
template <typename T>
__global__ void SharedMemoryBatchedGemm(const T* A, const T* B, T* C, int N,
                                        int batch_size);

#endif  // CODE_H
