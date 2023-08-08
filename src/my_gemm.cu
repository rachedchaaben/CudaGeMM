#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "my_gemm.h"
using namespace std;

#define CEIL(M, N) (((M) + (N)-1) / (N)) 

const int TILE_SIZE = 32;
const int TH_tile= 8; 
const int SM_tile = 128;


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
__global__ void globalMemorySgemm(const float* A, const float* B, float* C, int N)
{
    const int row = blockIdx.x * 32 + (threadIdx.x / 32);
    const int col = blockIdx.y * 32 + (threadIdx.x % 32);

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication kernel using global memory
__global__ void globalMemoryDgemm(const double* A, const double* B, double* C, int N)
{
    const int row = blockIdx.x * 32 + (threadIdx.x / 32);
    const int col = blockIdx.y * 32 + (threadIdx.x % 32);

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


// Single precision shared memory GeMM 
__global__ void SharedMemorySgemm(const float* A, const float* B, float* C, int N)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = SM_tile / TH_tile;
    int block_col_thread = SM_tile / TH_tile;
    int thread_num = block_row_thread * block_col_thread; 

    int tx = (threadIdx.x % block_row_thread) * TH_tile;
    int ty = (threadIdx.x / block_row_thread) * TH_tile;

    __shared__ float As[SM_tile * TH_tile];
    __shared__ float Bs[TH_tile * SM_tile];

    A = &A[by * SM_tile * N];
    B = &B[bx * SM_tile];
    C = &C[by * SM_tile * N + bx * SM_tile];

    int a_tile_row = threadIdx.x / TH_tile;
    int a_tile_col = threadIdx.x % TH_tile;
    int a_tile_stride = thread_num / TH_tile;

    int b_tile_row = threadIdx.x / SM_tile;
    int b_tile_col = threadIdx.x % SM_tile;
    int b_tile_stride = thread_num / SM_tile;

    float tmp[TH_tile][TH_tile] = {0.}; 
    #pragma unroll
    for (int k = 0; k < N; k += TH_tile) {
        #pragma unroll  
        for (int i = 0; i < SM_tile; i += a_tile_stride) {
            if(((by * SM_tile +(a_tile_row + i)) < N) && (a_tile_col < N))
                As[(a_tile_row + i) * TH_tile + a_tile_col] = A[(a_tile_row + i) * N + a_tile_col];
            else 
                As[(a_tile_row + i) * TH_tile + a_tile_col] = 0;
        }
        #pragma unroll
        for (int i = 0; i < TH_tile; i += b_tile_stride) {
            if (((b_tile_row + i) <N) && ((bx * SM_tile + b_tile_col )< N))
                Bs[(b_tile_row + i) * SM_tile + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
            else 
                Bs[(b_tile_row + i) * SM_tile + b_tile_col] =  0;

        }
        __syncthreads();
        A += TH_tile;
        B += TH_tile * N;
        #pragma unroll
        for (int i = 0; i < TH_tile; i++) {
            #pragma unroll  
            for (int j = 0; j < TH_tile; j++) {
                for (int l = 0; l < TH_tile; l++)
                    tmp[j][l] += As[(ty + j) * TH_tile + i] * Bs[tx + l + i * SM_tile];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TH_tile; j++) {
        for (int l = 0; l < TH_tile; l++)
            if (((by * SM_tile +ty +j)<N) && ((bx * SM_tile +tx+ l)<N))
            C[(ty + j) * N + tx + l] = tmp[j][l];
    }
}



// Double precision shared memory GeMM 
__global__ void SharedMemoryDgemm(const double* A, const double* B, double* C, int N)
{
    int row = (threadIdx.x / TILE_SIZE);
    int col = (threadIdx.x % TILE_SIZE);

    int A_row = blockIdx.x * TILE_SIZE + row;
    int B_col = blockIdx.y * TILE_SIZE + col;
    int C_sub_offset = A_row * N + B_col;

    double sum = 0.0;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    #pragma unroll
    for (int t = 0; t < numTiles; ++t) {
        __shared__ double shared_A[TILE_SIZE * TILE_SIZE];
        __shared__ double shared_B[TILE_SIZE * TILE_SIZE];

        int A_col = t * TILE_SIZE + col;
        int B_row = t * TILE_SIZE + row;

        if (A_row < N && A_col < N) {
            shared_A[row * TILE_SIZE +col] = A[A_row * N + A_col];
        } else {
            shared_A[row * TILE_SIZE +col] = 0.0;
        }

        if (B_row < N && B_col < N) {
            shared_B[row * TILE_SIZE +col] = B[B_row * N + B_col];
        } else {
            shared_B[row * TILE_SIZE +col] = 0.0;
        }

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[row * TILE_SIZE +k] * shared_B[k * TILE_SIZE +col];
        }

        __syncthreads();
    }

    if (A_row < N && B_col < N) {
        C[C_sub_offset] = sum;
    }
}

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

}
template __global__ void SharedMemoryBatchedGemm(const double* A,
                                                 const double* B, double* C,
                                                 int N, int batch_size);
template __global__ void SharedMemoryBatchedGemm(const float* A, const float* B,
                                                 float* C, int N,
                                                 int batch_size);