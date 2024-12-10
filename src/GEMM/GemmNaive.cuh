#ifndef HELLOCUDA_SRC_GEMM_GEMM_NAIVE_H
#define HELLOCUDA_SRC_GEMM_GEMM_NAIVE_H

/*
Implement Naive GEMM in the form C = α*(A@B)+β*C.
A = M x K row major
B = K x N row major
C = M x N row major
*/
__global__ void gemmNaive(int M, int N, int K, float alpha, const float *A,
                          const float *B, float beta, float *C);

#endif // HELLOCUDA_SRC_GEMM_GEMM_NAIVE_H