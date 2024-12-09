#include "src/Utils/Utils.cuh"
#include <algorithm>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

//===----------------------------------------------------------------------===//
// CUDA Error Check
//===----------------------------------------------------------------------===//

void check(cudaError_t err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//===----------------------------------------------------------------------===//
// Device Memory Management
//===----------------------------------------------------------------------===//

Device::~Device() {
  for (auto ptr : deviceAllocatedMemory) {
    if (ptr)
      CHECK_CUDA_ERROR(cudaFree(ptr));
  }
  deviceAllocatedMemory.clear();
}

Device::Device(Device &&other) noexcept {
  deviceAllocatedMemory = std::move(other.deviceAllocatedMemory);
  other.deviceAllocatedMemory.clear();
}

Device &Device::operator=(Device &&other) noexcept {
  if (this != &other) {
    for (auto ptr : deviceAllocatedMemory)
      CHECK_CUDA_ERROR(cudaFree(ptr));
    deviceAllocatedMemory = std::move(other.deviceAllocatedMemory);
    other.deviceAllocatedMemory.clear();
  }
  return *this;
}

void Device::set(int64_t id) { CHECK_CUDA_ERROR(cudaSetDevice(id)); }

void Device::copyFromHost(void *src, void *dst, int64_t bytes) {
  CHECK_CUDA_ERROR(cudaMemcpy(src, dst, bytes, cudaMemcpyHostToDevice));
}

void Device::copyToHost(void *dst, void *src, int64_t bytes) {
  CHECK_CUDA_ERROR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

//===----------------------------------------------------------------------===//
// Host Memory Management
//===----------------------------------------------------------------------===//

Host::~Host() {
  for (auto ptr : hostAllocatedMemory) {
    if (ptr)
      CHECK_CUDA_ERROR(cudaFreeHost(ptr));
  }
  hostAllocatedMemory.clear();
  for (auto ptr : deviceAllocatedMemory) {
    if (ptr)
      CHECK_CUDA_ERROR(cudaFree(ptr));
  }
  deviceAllocatedMemory.clear();
}

Host::Host(Host &&other) noexcept {
  hostAllocatedMemory = std::move(other.hostAllocatedMemory);
  other.hostAllocatedMemory.clear();
  deviceAllocatedMemory = std::move(other.deviceAllocatedMemory);
  other.deviceAllocatedMemory.clear();
}

Host &Host::operator=(Host &&other) noexcept {
  if (this != &other) {
    for (auto ptr : hostAllocatedMemory)
      CHECK_CUDA_ERROR(cudaFreeHost(ptr));
    hostAllocatedMemory = std::move(other.hostAllocatedMemory);
    other.hostAllocatedMemory.clear();
    for (auto ptr : deviceAllocatedMemory)
      CHECK_CUDA_ERROR(cudaFree(ptr));
    deviceAllocatedMemory = std::move(other.deviceAllocatedMemory);
    other.deviceAllocatedMemory.clear();
  }
  return *this;
}

//===----------------------------------------------------------------------===//
// Matrix Fill Operations
//===----------------------------------------------------------------------===//

void fillRandom(float *mat, int64_t numElements) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int i = 0; i < numElements; i++) {
    float rn = dist(gen);
    mat[i] = rn;
  }
}

void fillConstant(float *mat, int64_t numElements, float constant) {
  std::fill(mat, mat + numElements, constant);
}

void fillLinspace(float *mat, float start, int64_t size, float step) {
  int64_t writeIdx = 0;
  while (size > 0) {
    mat[writeIdx] = start;
    start += step;
    writeIdx += 1;
    size -= 1;
  }
}

void printMatrix(const float *mat, int m, int n) {
  int i;
  printf("[");
  for (i = 0; i < m * n; i++) {
    if ((i + 1) % n == 0)
      printf("%5.2f ", mat[i]);
    else
      printf("%5.2f, ", mat[i]);
    if ((i + 1) % n == 0) {
      if (i + 1 < m * n)
        printf(";\n");
    }
  }
  printf("]\n");
}

int add_two(int a, int b) { return a + b; }