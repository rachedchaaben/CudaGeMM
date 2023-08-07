#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

// Function to compare the results of two square matrices
template <typename T>
bool compareMatrices(T* A, T* B, int N, int batch_size);

// Funtion that generates Random square matrix in double precision
double* generateRandomMatrixDouble(int N, int M);

// Funtion that generates Random batch matrices in double precision
double* generateBatchRandomMatricesDouble(int N, int M, int BATCH_SIZE);

#endif  // MATRIX_GENERATION_H
