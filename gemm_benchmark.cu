// main.cu
#include <cuda_runtime.h>
#include <iostream>
#include "device_launch_parameters.h"
#include "kernels_calls.h"
#include "matrix.h"

const int MIN_MATRIX_SIZE = 1000;
const int MAX_MATRIX_SIZE = 1000;

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

        // Initialize input matrices with random values
        double* h_A = new double[N * N];
        double* h_B = new double[N * N];

        float* h_A_float = new float[N * N];
        float* h_B_float = new float[N * N];
        h_A = generateRandomMatrixDouble(N, N);
                for (int i = 0; i < N * N; ++i) {
            h_A_float[i] = static_cast<float>(h_A[i]);
        }
        
        h_B = generateRandomMatrixDouble(N, N);
        for (int i = 0; i < N * N; ++i) {
            h_B_float[i] = static_cast<float>(h_B[i]);
        }

        double *h_C1, *h_C2, *h_C3, *h_C4;
        float *h_C1_float, *h_C2_float, *h_C3_float, *h_C4_float;

        h_C1 = call_cublasGemm(h_A, h_B, N);
        h_C1_float = call_cublasGemm(h_A_float, h_B_float, N);

        h_C2_float = call_cutlassSgemm(h_A_float, h_B_float, N);

        h_C3 = call_DgemmGlobalMemory(h_A, h_B, N);
        h_C3_float = call_SgemmGlobalMemory(h_A_float, h_B_float, N);

        h_C4 = call_DgemmSharedMemory(h_A, h_B, N);
        h_C4_float = call_SgemmSharedMemory(h_A_float, h_B_float, N);


        // Compare results of CUDA kernels
        if (compareMatrices(h_C1, h_C4, N, 1) &&
        //    compareMatrices(h_C3, h_C1, N, 1) &&
            compareMatrices(h_C4, h_C3, N, 1)) {
            std::cout << "Correct Double Precison CUDA Kernels\n";
        } else
            std::cerr
                << "Double Precison CUDA Kernels produced inconsistent results!"
                << std::endl;

        if (compareMatrices(h_C1_float, h_C2_float, N, 1) &&
            compareMatrices(h_C3_float, h_C1_float, N, 1) &&
            compareMatrices(h_C2_float, h_C4_float, N, 1)) {
            std::cout << "Correct Float Precison CUDA Kernels\n";
        } else
            std::cerr
                << "Float Precison CUDA Kernels produced inconsistent results!"
                << std::endl;
        std::cout << std::endl;
    }
    return 0;
}