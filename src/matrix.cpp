#include "matrix.h"
#include <iostream>
#include <random>


// Function to compare the results of two square matrices
template <typename T>
bool compareMatrices(T* A, T* B, int N, int batch_size)
{
    for (int i = 0; i < N * N * batch_size; ++i) {
        if (std::abs(A[i] - B[i]) > 1e-3) {
            return false;
        }
    }
    return true;
}
template bool compareMatrices(float* A, float* B, int N, int batch_size);
template bool compareMatrices(double* A, double* B, int N, int batch_size);


// Funtion that generates Random matrix in double precision
double* generateRandomMatrixDouble(int N, int M)
{
    double* matrix = new double[N * N];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            matrix[i * M + j] = dis(gen);
        }
    }

    return matrix;
}

// Funtion that generates Random batch matrices in double precision
double* generateBatchRandomMatricesDouble(int N, int M, int BATCH_SIZE)
{
    double* matrix = new double[N * N * BATCH_SIZE];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int k = 0; k < BATCH_SIZE; ++k) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                matrix[i + j * BATCH_SIZE * M + k * M] = dis(gen);
            }
        }
    }
    return matrix;
}
