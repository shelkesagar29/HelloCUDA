#include "src/Utils/Utils.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

__global__ void pointwiseAddNaive(const float *A, const float *B, float *C,
                                  int rows, int cols) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    int index = row * cols + col;
    C[index] = A[index] + B[index];
  }
}

int main() {
  int64_t m{2048};
  int64_t n{2048};

  float *A = nullptr, *B = nullptr, *C = nullptr;    // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr; // device matrices

  A = (float *)malloc(sizeof(float) * m * n);
  B = (float *)malloc(sizeof(float) * m * n);
  C = (float *)malloc(sizeof(float) * m * n);

  fillRandom(A, m * n);
  fillRandom(B, m * n);
  fillConstant(C, m * n, 0);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&dA, sizeof(float) * m * n));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dB, sizeof(float) * m * n));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&dC, sizeof(float) * m * n));

  CHECK_CUDA_ERROR(
      cudaMemcpy(dA, A, sizeof(float) * m * n, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dB, B, sizeof(float) * m * n, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dC, C, sizeof(float) * m * n, cudaMemcpyHostToDevice));

  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(m, 32), CEIL_DIV(n, 32));

  cudaEvent_t start;
  cudaEvent_t end;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&end));

  // Warm up
  for (int64_t i = 0; i < 50; i++) {
    pointwiseAddNaive<<<gridDim, blockDim>>>(dA, dB, dC, m, n);
  }
  // Benchmark run
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  pointwiseAddNaive<<<gridDim, blockDim>>>(dA, dB, dC, m, n);
  CHECK_CUDA_ERROR(cudaEventRecord(end));
  cudaEventSynchronize(start);
  cudaEventSynchronize(end);
  float executionTimeMs{0.0};
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&executionTimeMs, start, end));
  printf("Runtime: %f ms \n", executionTimeMs);
  CHECK_CUDA_ERROR(
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

  free(A);
  free(B);
  free(C);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}