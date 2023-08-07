#include "wrapper.h"

// Template specialization for float
template <>
void cublasGemm_w<float>(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, int m, int n, int k,
                         const float* alpha, const float* A, int lda,
                         const float* B, int ldb, const float* beta, float* C,
                         int ldc)
{
    cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                ldc);
}


// Template specialization for double
template <>
void cublasGemm_w<double>(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const double* alpha, const double* A, int lda,
                          const double* B, int ldb, const double* beta,
                          double* C, int ldc)
{
    cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                ldc);
}
// Template specialization for float
template <>
void cublasBatchedGemm_w<float>(cublasHandle_t handle, cublasOperation_t transa,
                                cublasOperation_t transb, int m, int n, int k,
                                const float* alpha, const float* Aarray[],
                                int lda, const float* Barray[], int ldb,
                                const float* beta, float* Carray[], int ldc,
                                int batchCount)
{
    cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                       Barray, ldb, beta, Carray, ldc, batchCount);
}


template <>
void cublasBatchedGemm_w<double>(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb, int m, int n, int k,
                                 const double* alpha, const double* Aarray[],
                                 int lda, const double* Barray[], int ldb,
                                 const double* beta, double* Carray[], int ldc,
                                 int batchCount)
{
    cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                       Barray, ldb, beta, Carray, ldc, batchCount);
}
