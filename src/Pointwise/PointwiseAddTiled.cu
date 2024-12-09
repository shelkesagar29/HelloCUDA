#include "src/Utils/Utils.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void pointwiseAddTiled(const float *A, const float *B, float *C,
                                  int rows, int cols) {
  // RTX 4090 TI has 48KB shared memory per thread block
  __shared__ float tileA[TILE_SIZE][TILE_SIZE]; // uses 16*16*8(sizeof(float)) =
                                                // 2048 bytes of memory = 2 KB
  __shared__ float tileB[TILE_SIZE][TILE_SIZE]; // uses 2KB memory

  // While launching, size of thread block is same as TILE_SIZE.
  // We could also use good old `bockDim.x/.y` instead of TILE_SIZE
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  if (row < rows && col < cols) {
    tileA[threadIdx.y][threadIdx.x] = A[row * cols + col];
    tileB[threadIdx.y][threadIdx.x] = B[row * cols + col];
  } else {
    tileA[threadIdx.y][threadIdx.x] = 0.0f;
    tileB[threadIdx.y][threadIdx.x] = 0.0f;
  }

  // Wait for all threads to load into tile
  __syncthreads();
  if (row < rows && col < cols) {
    C[row * cols + col] =
        tileA[threadIdx.y][threadIdx.x] + tileB[threadIdx.y][threadIdx.x];
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

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim(CEIL_DIV(m, TILE_SIZE), CEIL_DIV(n, TILE_SIZE));

  cudaEvent_t start;
  cudaEvent_t end;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&end));

  // Warm up
  for (int64_t i = 0; i < 50; i++) {
    pointwiseAddTiled<<<gridDim, blockDim>>>(dA, dB, dC, m, n);
  }
  // Benchmark run
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  pointwiseAddTiled<<<gridDim, blockDim>>>(dA, dB, dC, m, n);
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