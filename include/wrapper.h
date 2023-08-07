#ifndef GEMM_WRAPPER_H
#define GEMM_WRAPPER_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Template function to call the appropriate cublas gemm function based on the
// template type
template <typename T>
void cublasGemm_w(cublasHandle_t handle, cublasOperation_t transa,
                  cublasOperation_t transb, int m, int n, int k, const T* alpha,
                  const T* A, int lda, const T* B, int ldb, const T* beta, T* C,
                  int ldc);

// Template specialization for T
template <typename T>
void cublasBatchedGemm_w(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, int m, int n, int k,
                         const T* alpha, const T* Aarray[], int lda,
                         const T* Barray[], int ldb, const T* beta, T* Carray[],
                         int ldc, int batchCount);


#endif  // GEMM_WRAPPER_H
