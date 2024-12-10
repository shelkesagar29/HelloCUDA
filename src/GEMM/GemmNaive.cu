#include "src/GEMM/GemmNaive.cuh"
#include "src/Utils/Utils.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

__global__ void gemmNaive(int M, int N, int K, float alpha, const float *A,
                          const float *B, float beta, float *C) {

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[row * K + i] * B[i * N + col];
    }
    // C = α*(A@B)+β*C
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
}

int main() {
  int64_t m{3};
  int64_t k{3};
  int64_t n{3};

  float *A = nullptr, *B = nullptr, *C = nullptr;    // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * m * k);
  B = (float *)malloc(sizeof(float) * k * n);
  C = (float *)malloc(sizeof(float) * m * n);

  fillLinspace(A, 1.0f, m * k, 1.0f);
  fillLinspace(B, 1.0f, k * n, 1.0f);
  fillConstant(C, m * n, 0);
  printMatrix(A, m, k);
  printMatrix(B, k, n);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&dA, sizeof(float) * m * k));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dB, sizeof(float) * k * n));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dC, sizeof(float) * m * n));

  CHECK_CUDA_ERROR(
      cudaMemcpy(dA, A, sizeof(float) * m * k, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dB, B, sizeof(float) * k * n, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dC, C, sizeof(float) * m * n, cudaMemcpyHostToDevice));

  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
  float alpha{1.0f};
  float beta{1.0f};

  gemmNaive<<<gridDim, blockDim>>>(m, n, k, alpha, dA, dB, beta, dC);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

  printMatrix(C, m, n);
  free(A);
  free(B);
  free(C);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}