// batched_gemm_benchmark.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "device_launch_parameters.h"
#include "kernels_calls.h"
#include "matrix.h"

const int MIN_BATCH_SIZE = 100;
const int MAX_BATCH_SIZE = 400;

const int MIN_MATRIX_SIZE = 100;
const int MAX_MATRIX_SIZE = 400;

int main()
{
    // Checking for Nvidia device availability
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 1;
    }

    int selectedDevice = 0;
    cudaSetDevice(selectedDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, selectedDevice);

    printf("Device Name: %s\n", deviceProp.name);

    if (deviceProp.integrated) {
        printf("The code is running on an INTEGRATED GPU.\n");
    } else {
        printf("The code is running on a DEDICATED NVIDIA GPU.\n");
    };

    // Loop through different matrix sizes
    for (int N = MIN_MATRIX_SIZE; N <= MAX_MATRIX_SIZE; N += 50) {
        std::cout << "Matrix Size: " << N << " x " << N <<", Batch size : " << N << std::endl;
        int BATCH_SIZE =N;
        double* h_A = new double[N * N * BATCH_SIZE];
        double* h_B = new double[N * N * BATCH_SIZE];

        float* h_A_float = new float[N * N * BATCH_SIZE];
        float* h_B_float = new float[N * N * BATCH_SIZE];

        h_A = generateBatchRandomMatricesDouble(N, N, BATCH_SIZE);
        for (int i = 0; i < N * N * BATCH_SIZE; ++i) {
            h_A_float[i] = static_cast<float>(h_A[i]);
        }

        h_B = generateBatchRandomMatricesDouble(N, N, BATCH_SIZE);
        for (int i = 0; i < N * N * BATCH_SIZE; ++i) {
            h_B_float[i] = static_cast<float>(h_B[i]);
        }
        double *h_C1, *h_C2, *h_C3, *h_C4;
        float *h_C1_float, *h_C2_float, *h_C3_float, *h_C4_float;

        h_C1 = call_cublasGemmBatched(h_A, h_B, N, BATCH_SIZE);
        h_C1_float = call_cublasGemmBatched(h_A_float, h_B_float, N, BATCH_SIZE);

        h_C2 = call_CutlassBatchedGemm(h_A, h_B, N, BATCH_SIZE);
        h_C2_float = call_CutlassBatchedGemm(h_A_float, h_B_float, N, BATCH_SIZE);

        h_C3 = call_BatchedGemmGlobalMemory(h_A, h_B, N, BATCH_SIZE);
        h_C3_float =
            call_BatchedGemmGlobalMemory(h_A_float, h_B_float, N, BATCH_SIZE);

        //  h_C4=call_BatchedGemmSharedMemory(h_A,h_B,N,BATCH_SIZE);
        // h_C4_float=call_BatchedGemmSharedMemory(h_A_float,h_B_float,N,BATCH_SIZE);

        if (compareMatrices(h_C1, h_C2, N, BATCH_SIZE) &&
            compareMatrices(h_C3, h_C1, N, BATCH_SIZE) &&
            compareMatrices(h_C2, h_C3, N, BATCH_SIZE)) {
            std::cout << "Correct Double Precison CUDA Kernels\n";
        } else
            std::cerr
                << "Double Precison CUDA Kernels produced inconsistent results!"
                << std::endl;

        if (compareMatrices(h_C1_float, h_C2_float, N, BATCH_SIZE) &&
            compareMatrices(h_C3_float, h_C1_float, N, BATCH_SIZE) &&
            compareMatrices(h_C2_float, h_C3_float, N, BATCH_SIZE)) {
            std::cout << "Correct Float Precison CUDA Kernels\n";
        } else
            std::cerr
                << "Float Precison CUDA Kernels produced inconsistent results!"
                << std::endl;
    }
    return 0;
}