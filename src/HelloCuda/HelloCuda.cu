#include <cstdio>
#include <cuda_runtime_api.h>

__global__ void cuda_hello() { printf("Hello CUDA!\n"); }

int main() {

  cuda_hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}