// main.cu
#include <cuda_runtime.h>
#include <iostream>
#include "device_launch_parameters.h"
#include "kernels_calls.h"
#include "matrix.h"

const int MIN_MATRIX_SIZE = 1000;
const int MAX_MATRIX_SIZE = 4000;

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

    // Loop through different matrix sizes and compare the performance
    for (int N = MIN_MATRIX_SIZE; N <= MAX_MATRIX_SIZE; N += 500) {
        std::cout << "Matrix Size: " << N << " x " << N << std::endl;

        // Allocate memory for host matrices
        double* h_A = new double[N * N];
        double* h_B = new double[N * N];
        double* h_C_cuda_global = new double[N * N];
        double* h_C_cuda_shared = new double[N * N];
        double* h_C_cublas = new double[N * N];
        float* h_C_cublas_float = new float[N * N];
        float* h_A_float = new float[N * N];
        float* h_B_float = new float[N * N];
        float* h_C_cuda_global_float = new float[N * N];
        float* h_C_cuda_shared_float = new float[N * N];


        // Initialize input matrices with random values
        h_A = generateRandomMatrixDouble(N, N);
        h_B = generateRandomMatrixDouble(N, N);
        for (int i = 0; i < N * N; ++i) {
            h_A_float[i] = static_cast<float>(h_A[i]);
        }
        for (int i = 0; i < N * N; ++i) {
            h_B_float[i] = static_cast<float>(h_B[i]);
        }

        // cublas gemm as reference for output verification
        h_C_cublas = call_cublasGemm(h_A, h_B, N);
        h_C_cublas_float = call_cublasGemm(h_A_float, h_B_float, N);

        h_C_cuda_global = call_DgemmGlobalMemory(h_A, h_B, N);
        h_C_cuda_global_float = call_SgemmGlobalMemory(h_A_float, h_B_float, N);

        h_C_cuda_shared = call_DgemmSharedMemory(h_A, h_B, N);
        h_C_cuda_shared_float = call_SgemmSharedMemory(h_A_float, h_B_float, N);

                float* h_CPU = new float [N * N];
                h_CPU = gemm_cpu(h_A_float, h_B_float,N,N,N);
        /**/
        // Compare results of CUDA kernels
        if (compareMatrices(h_C_cublas, h_C_cuda_global, N, 1) &&
            compareMatrices(h_C_cublas, h_C_cuda_shared, N, 1)) {
            std::cout << "Correct Double Precison CUDA Kernels\n";
        } else
            std::cerr
                << "Double Precison CUDA Kernels produced inconsistent results!"
                << std::endl;

        if (compareMatrices(h_C_cublas_float, h_C_cuda_shared_float, N, 1) &&
            compareMatrices(
                h_C_cuda_global_float, h_C_cublas_float, N,
                1) /*&& compareMatrices(h_CPU,h_C_cuda_shared_float,N)/**/) {
            std::cout << "Correct Single Precision CUDA kernels\n";
        } else
            std::cerr
                << "Single Precison CUDA Kernels produced inconsistent results!"
                << std::endl;

        std::cout << std::endl;
    }
    return 0;
}