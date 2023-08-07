#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "my_gemm.h"
using namespace std;

const int TILE_SIZE = 32;


// reference CPU GeMM to check for correctness
float* gemm_cpu(float* A, float* B, int m, int n, int k)
{
    float* C = new float[n * m];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    return C;
}

// Matrix multiplication kernel using global memory
template <typename T>
__global__ void globalMemoryGemm(const T* A, const T* B, T* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        T sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
template __global__ void globalMemoryGemm(const double* A, const double* B,
                                          double* C, int N);
template __global__ void globalMemoryGemm(const float* A, const float* B,
                                          float* C, int N);


// Matrix multiplication kernel using shared memory
template <typename T>
__global__ void SharedMemoryGemm(const T* A, const T* B, T* C, int N)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    int A_row = blockIdx.y * blockDim.y + row;
    int B_col = blockIdx.x * blockDim.x + col;
    int C_sub_offset = A_row * N + B_col;

    T sum = 0.0;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        __shared__ T shared_A[TILE_SIZE][TILE_SIZE];
        __shared__ T shared_B[TILE_SIZE][TILE_SIZE];

        int A_col = t * TILE_SIZE + col;
        int B_row = t * TILE_SIZE + row;

        if (A_row < N && A_col < N) {
            shared_A[row][col] = A[A_row * N + A_col];
        } else {
            shared_A[row][col] = 0.0;
        }

        if (B_row < N && B_col < N) {
            shared_B[row][col] = B[B_row * N + B_col];
        } else {
            shared_B[row][col] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[row][k] * shared_B[k][col];
        }

        __syncthreads();
    }

    if (A_row < N && B_col < N) {
        C[C_sub_offset] = sum;
    }
}
template __global__ void SharedMemoryGemm(const double* A, const double* B,
                                          double* C, int N);
template __global__ void SharedMemoryGemm(const float* A, const float* B,
                                          float* C, int N);

// Kernel to perform batched Gemm
template <typename T>
__global__ void GlobalMemoryBatchedGemm(const T* A, const T* B, T* C, int N,
                                        int batch_size)
{
    int B_col = blockIdx.y * blockDim.y + threadIdx.y;
    int A_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (A_row < N && B_col < N * batch_size) {
        T sum = 0.0;
        int col = B_col % N;
        int batch_idx = B_col / N;

        for (int k = 0; k < N; ++k) {
            sum += A[A_row * N * batch_size + k + batch_idx * N] *
                   B[k * N * batch_size + col + batch_idx * N];
        }
        C[A_row * N * batch_size + col + batch_idx * N] = sum;
    }
}
template __global__ void GlobalMemoryBatchedGemm(const double* A,
                                                 const double* B, double* C,
                                                 int N, int batch_size);
template __global__ void GlobalMemoryBatchedGemm(const float* A, const float* B,
                                                 float* C, int N,
                                                 int batch_size);


// Kernel to perform batched Gemm with shared memory optimization
template <typename T>
__global__ void SharedMemoryBatchedGemm(const T* A, const T* B, T* C, int N,
                                        int batch_size)
{
    int B_col = blockIdx.y * blockDim.y + threadIdx.y;
    int A_row = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for tiles of matrices A and B
    __shared__ T tileA[TILE_SIZE][TILE_SIZE];
    __shared__ T tileB[TILE_SIZE][TILE_SIZE];

    T sum = 0.0;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int col = B_col % N;

        // Calculate the starting index of the batch in global memory
        int A_batch_start = A_row * N * batch_size + batch_idx * N;
        int B_batch_start = batch_idx * N + col;

        // Load tiles of matrices A and B into shared memory
        tileA[threadIdx.y][threadIdx.x] =
            A[A_batch_start + threadIdx.y * N + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] =
            B[B_batch_start + threadIdx.y * N + threadIdx.x];

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute the dot product of the tiles
        for (int k = 0; k < N; k += TILE_SIZE) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize to make sure all threads are done using the tiles
        __syncthreads();
    }

    // Write the result to global memory
    if (A_row < N && B_col < N * batch_size) {
        C[A_row * N * batch_size + B_col] = sum;
    }


    /**/


    /* int col = threadIdx.y;
     int row = threadIdx.x;

     int B_col = blockIdx.y * blockDim.y + col;
     int A_row = blockIdx.x * blockDim.x + row;

     int batch_col = B_col%N;
     int batch_idx = B_col/N;

     int C_sub_offset = A_row * N *batch_size + B_col;

     T sum = 0.0;
     int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
     for (int t = 0; t < numTiles; ++t) {
         __shared__ T shared_A[TILE_SIZE][TILE_SIZE];
         __shared__ T shared_B[TILE_SIZE][TILE_SIZE];

         int A_col = t * TILE_SIZE + col;
         int B_row = t * TILE_SIZE + row;

         if (A_row < N && A_col < N && batch_idx <batch_size) {
             shared_A[row][col] = A[A_row * N *batch_size+ A_col +batch_idx*N];
         } else {
             shared_A[row][col] = 0.0;
         }

         if (B_row < N && batch_col < N && batch_idx <batch_size) {
             shared_B[row][col] = B[B_row * N *batch_size + B_col];
         } else {
             shared_B[row][col] = 0.0;
         }

         __syncthreads();

         for (int k = 0; k < TILE_SIZE; ++k) {
             sum += shared_A[row][k] * shared_B[k][col];
         }

         __syncthreads();
     }

     if (A_row < N && batch_col < N && batch_idx < batch_size) {
         C[C_sub_offset] = sum;
     }/**/
}
template __global__ void SharedMemoryBatchedGemm(const double* A,
                                                 const double* B, double* C,
                                                 int N, int batch_size);
template __global__ void SharedMemoryBatchedGemm(const float* A, const float* B,
                                                 float* C, int N,
                                                 int batch_size);