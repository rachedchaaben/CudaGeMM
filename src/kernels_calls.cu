#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/layout/matrix.h"
#include "kernels_calls.h"
#include "wrapper.h"
#define CEIL(M, N) (((M) + (N)-1) / (N)) 

using namespace std;

// cublas matrix multiplication kernel T precision
template <typename T>
T* call_cublasGemm(const T* h_A, const T* h_B, int N)
{
    // handle creation for cublas gemm
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on GPU
    T* d_A;
    T* d_B;
    T* d_C;
    T* h_C = new T[N * N];

    cudaMalloc(&d_A, N * N * sizeof(T));
    cudaMalloc(&d_B, N * N * sizeof(T));
    cudaMalloc((void**)&d_C, N * N * sizeof(T));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(T), cudaMemcpyHostToDevice);

    const T alpha = 1.0;
    const T beta = 0.0;
    cublasGemm_w(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A,
                 N, &beta, d_C, N);
    cudaMemcpy(h_C, d_C, N * N * sizeof(T), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}
template double* call_cublasGemm(const double* h_A, const double* h_B, int N);
template float* call_cublasGemm(const float* h_A, const float* h_B, int N);

double* call_DgemmGlobalMemory(const double* h_A, const double* h_B, int N)
{
    double* h_C = new double[N * N];

    // Allocate memory on GPU
    double* d_A;
    double* d_B;
    double* d_C;

    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, N * N * sizeof(double));
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate the grid and block dimensions for CUDA kernels
    dim3 block(32 * 32);
    dim3 grid(CEIL(N,32), CEIL(N,32));

    // Call the global memory double precision kernel
    globalMemoryDgemm<<<grid, block>>>(d_A, d_B, d_C, N);

    // Copy results back from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}

float* call_SgemmGlobalMemory(const float* h_A, const float* h_B, int N)
{
    float* h_C = new float[N * N];

    // Allocate memory on GPU
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the grid and block dimensions for CUDA kernels
    dim3 block(32 * 32);
    dim3 grid(CEIL(N,32), CEIL(N,32));

    // Call the global memory float precision kernel
    globalMemorySgemm<<<grid, block>>>(d_A, d_B, d_C, N);

    // Copy results back from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}

double* call_DgemmSharedMemory(const double* h_A, const double* h_B, int N)
{
    double* h_C = new double[N * N];

    // Allocate memory on GPU
    double* d_A;
    double* d_B;
    double* d_C;

    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, N * N * sizeof(double));
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate the grid and block dimensions for CUDA kernels
    dim3 block(32 * 32);
    dim3 grid(CEIL(N,32), CEIL(N,32));

    // Call the global memory double precision kernel
    SharedMemoryDgemm<<<grid, block>>>(d_A, d_B, d_C, N);

    // Copy results back from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}

float* call_SgemmSharedMemory(const float* h_A, const float* h_B, int N)
{
    float* h_C = new float[N * N];

    // Allocate memory on GPU
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the grid and block dimensions for CUDA kernels

    dim3 block(256);
    dim3 grid(CEIL(N,128), CEIL(N,128));


    // Call the global memory float precision kernel
    SharedMemorySgemm<<<grid, block>>>(d_A, d_B, d_C, N);

    // Copy results back from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}

// cutlass matrix multiplication kernel T precision
float* call_cutlassSgemm(const float* h_A, const float* h_B, int N)
{
    float* h_C = new float[N * N];

    // Allocate memory on GPU
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm =
        cutlass::gemm::device::Gemm<float,         // Data-type of A matrix
                                    ColumnMajor,   // Layout of A matrix
                                    float,         // Data-type of B matrix
                                    ColumnMajor,   // Layout of B matrix
                                    float,         // Data-type of C matrix
                                    ColumnMajor>;  // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args(
        {N ,N ,N},  // Gemm Problem dimensions
        {d_B, N}, 
        {d_A, N},   // Tensor-ref for source matrix A
        {d_C, N},   // Tensor-ref for source matrix C
        {d_C, N},   // Tensor-ref for destination matrix D (may be different
                    // memory than source C matrix)
        {1.0f, 0.0f});  // Scalars used in the Epilogue
    
    cutlass::Status status = gemm_operator(args);

    // Copy results back from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}


// cublas Batched matrix multiplication kernel T precision
template <typename T>
T* call_cublasGemmBatched(const T* h_A, const T* h_B, int N, int BATCH_SIZE)
{
    T* h_C = new T[N * N * BATCH_SIZE];
    T *d_A, *d_B, *d_C;
    size_t size = sizeof(T) * N * N * BATCH_SIZE;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);

    T *A_array[BATCH_SIZE], *B_array[BATCH_SIZE];
    T* C_array[BATCH_SIZE];
    {
        for (int j = 0; j < BATCH_SIZE; ++j) {
            A_array[j] = d_A + j * N;
            B_array[j] = d_B + j * N;
            C_array[j] = d_C + j * N;
        }
    }
    const T **d_A_array, **d_B_array;
    T** d_C_array;
    cudaMalloc((void**)&d_A_array, BATCH_SIZE * sizeof(T*));
    cudaMalloc((void**)&d_B_array, BATCH_SIZE * sizeof(T*));
    cudaMalloc((void**)&d_C_array, BATCH_SIZE * sizeof(T*));
    cudaMemcpy(d_A_array, A_array, BATCH_SIZE * sizeof(T*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array, B_array, BATCH_SIZE * sizeof(T*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array, C_array, BATCH_SIZE * sizeof(T*),
               cudaMemcpyHostToDevice);

    const T alpha = 1.0;
    const T beta = 0.0;

    int lda = N * BATCH_SIZE;
    int ldb = N * BATCH_SIZE;
    int ldc = N * BATCH_SIZE;

    cublasBatchedGemm_w(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                        d_B_array, ldb, d_A_array, lda, &beta, d_C_array, ldc,
                        BATCH_SIZE);
    cublasDestroy(handle);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);

    return h_C;
}
template double* call_cublasGemmBatched(const double* h_A, const double* h_B,
                                        int N, int BATCH_SIZE);
template float* call_cublasGemmBatched(const float* h_A, const float* h_B,
                                       int N, int BATCH_SIZE);

//
template <typename T>
T* call_BatchedGemmGlobalMemory(const T* h_A, const T* h_B, int N,
                                int BATCH_SIZE)
{
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * BATCH_SIZE * sizeof(T));
    cudaMalloc((void**)&d_B, N * N * BATCH_SIZE * sizeof(T));
    cudaMalloc((void**)&d_C, N * N * BATCH_SIZE * sizeof(T));

    // Transfer input matrices from host to GPU
    cudaMemcpy(d_A, h_A, N * N * BATCH_SIZE * sizeof(T),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * BATCH_SIZE * sizeof(T),
               cudaMemcpyHostToDevice);

    dim3 block_dim(8, 128);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (N * BATCH_SIZE + block_dim.y - 1) / block_dim.y);

    T* h_C = new T[N * N * BATCH_SIZE];

    GlobalMemoryBatchedGemm<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N,
                                                     BATCH_SIZE);

    cudaDeviceSynchronize();
    // Transfer result matrix from GPU to host
    cudaMemcpy(h_C, d_C, N * N * BATCH_SIZE * sizeof(T),
               cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}
template double* call_BatchedGemmGlobalMemory(const double* h_A,
                                              const double* h_B, int N,
                                              int BATCH_SIZE);
template float* call_BatchedGemmGlobalMemory(const float* h_A, const float* h_B,
                                             int N, int BATCH_SIZE);

template <typename T>
T* call_BatchedGemmSharedMemory(const T* h_A, const T* h_B, int N,
                                int BATCH_SIZE)
{
    T *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * BATCH_SIZE * sizeof(T));
    cudaMalloc((void**)&d_B, N * N * BATCH_SIZE * sizeof(T));
    cudaMalloc((void**)&d_C, N * N * BATCH_SIZE * sizeof(T));

    // Transfer input matrices from host to GPU
    cudaMemcpy(d_A, h_A, N * N * BATCH_SIZE * sizeof(T),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * BATCH_SIZE * sizeof(T),
               cudaMemcpyHostToDevice);


    dim3 block_dim(32, 32);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (N * BATCH_SIZE + block_dim.y - 1) / block_dim.y);
    T* h_C = new T[N * N * BATCH_SIZE];

    SharedMemoryBatchedGemm<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N,
                                                     BATCH_SIZE);

    cudaDeviceSynchronize();
    // Transfer result matrix from GPU to host
    cudaMemcpy(h_C, d_C, N * N * BATCH_SIZE * sizeof(T),
               cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}
template double* call_BatchedGemmSharedMemory(const double* h_A,
                                              const double* h_B, int N,
                                              int BATCH_SIZE);
template float* call_BatchedGemmSharedMemory(const float* h_A, const float* h_B,
                                             int N, int BATCH_SIZE);


template <typename T>
T* call_CutlassBatchedGemm(const T* h_A, const T* h_B, int N, int BATCH_SIZE)
{
    T* A;
    T* B;
    T* C;
    T alpha = 1.0;
    T beta = 2.0;

    cudaMalloc(&A, N * N * BATCH_SIZE * sizeof(T));
    cudaMalloc(&B, N * N * BATCH_SIZE * sizeof(T));
    cudaMalloc(&C, N * N * BATCH_SIZE * sizeof(T));

    cudaMemcpy(A, h_A, N * N * BATCH_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, N * N * BATCH_SIZE * sizeof(T), cudaMemcpyHostToDevice);

    using Gemm =
        cutlass::gemm::device::GemmBatched<T, cutlass::layout::ColumnMajor, T,
                                           cutlass::layout::ColumnMajor, T,
                                           cutlass::layout::ColumnMajor>;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({{N, N, N},
                                      {B, N * BATCH_SIZE},
                                      N,
                                      {A, N * BATCH_SIZE},
                                      N,
                                      {C, N * BATCH_SIZE},
                                      N,
                                      {C, N * BATCH_SIZE},
                                      N,
                                      {alpha, beta},
                                      BATCH_SIZE});

    T* h_C = new T[N * N * BATCH_SIZE];

    cudaMemcpy(h_C, C, N * N * BATCH_SIZE * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);


    return h_C;
}
template double* call_CutlassBatchedGemm(const double* h_A, const double* h_B,
                                         int N, int BATCH_SIZE);
template float* call_CutlassBatchedGemm(const float* h_A, const float* h_B,
                                        int N, int BATCH_SIZE);