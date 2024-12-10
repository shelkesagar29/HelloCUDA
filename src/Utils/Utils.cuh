#ifndef HELLOCUDA_SRC_UTILS_UTILS_H
#define HELLOCUDA_SRC_UTILS_UTILS_H

#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

//===----------------------------------------------------------------------===//
// CUDA Error Check
//===----------------------------------------------------------------------===//

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
/// Prints error message and terminates program with sys.exit if `err` is not
/// `cudaSuccess`.
void check(cudaError_t err, const char *const func, const char *const file,
           const int line);

//===----------------------------------------------------------------------===//
// Device Memory Management
//===----------------------------------------------------------------------===//

// TODO:  set device
class Device {
public:
  Device() = default;
  ~Device();

  // Prevent copying to avoid double free
  Device(const Device &other) = delete;
  Device &operator=(const Device &other) = delete;

  Device(Device &&other) noexcept;
  Device &operator=(Device &&other) noexcept;

  /// Set device with `id` to be used for GPU execution.
  void set(int64_t id);

  /// Allocates `bytes` number of bytes on device and returns
  /// reinterpret casted pointer.
  template <typename T>
  T *allocate(int64_t bytes) {
    void *ptr;
    CHECK_CUDA_ERROR(cudaMalloc(&ptr, bytes));
    if (ptr)
      deviceAllocatedMemory.push_back(ptr);
    return reinterpret_cast<T *>(ptr);
  }

  /// Copies data from host to device.
  /// TODO: Add stream support
  void copyFromHost(void *src, void *dst, int64_t bytes);

  /// Copies data from device to host.
  /// TODO: Add stream support
  void copyToHost(void *dst, void *src, int64_t bytes);

private:
  std::vector<void *> deviceAllocatedMemory;
};

//===----------------------------------------------------------------------===//
// Host Memory Management
//===----------------------------------------------------------------------===//

class Host {
public:
  Host() = default;
  ~Host();

  // Prevent copying to avoid double free
  Host(const Host &other) = delete;
  Host &operator=(const Host &other) = delete;

  Host(Host &&other) noexcept;
  Host &operator=(Host &&other) noexcept;

  /// Allocates page locked memory of size `bytes` and return reinterpret casted
  /// pointer. Behavior is `cudaHostAlloc` with `cudaHostAllocDefault` flag,
  /// also same as `cudaMallocHost`.
  /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
  template <typename T>
  T *allocatePageLocked(int64_t bytes) {
    void *ptr;
    CHECK_CUDA_ERROR(cudaHostAlloc(&ptr, bytes, 0));
    if (ptr)
      hostAllocatedMemory.push_back(ptr);
    return reinterpret_cast<T *>(ptr);
  }

  /// Allocates zero copy memory of size `bytes` and return reinterpret casted
  /// pointer. Zero copy means data is not copied into the device buffer but
  /// instead accessed over PCIE whenever needed. In all other ways, either user
  /// or CUDA driver moves data into the device buffer. Behavior is
  /// `cudaHostAlloc` with `cudaHostAllocMapped` flag.
  /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
  template <typename T>
  T *allocateZeroCopy(int64_t bytes) {
    void *ptr;
    CHECK_CUDA_ERROR(cudaHostAlloc(&ptr, bytes, 2));
    if (ptr)
      hostAllocatedMemory.push_back(ptr);
    return reinterpret_cast<T *>(ptr);
  }

  /// Allocated unified memory of size `bytes` and return reinterpret casted
  /// pointer.
  /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b
  template <typename T>
  T *allocateUnified(int64_t bytes) {
    void *ptr;
    CHECK_CUDA_ERROR(cudaMallocManaged(&ptr, bytes));
    if (ptr)
      deviceAllocatedMemory.push_back(ptr);
    return reinterpret_cast<T *>(ptr);
  }

private:
  std::vector<void *> hostAllocatedMemory;
  // Holds unified memory pointer
  std::vector<void *> deviceAllocatedMemory;
};

//===----------------------------------------------------------------------===//
// Matrix Fill Operations
//===----------------------------------------------------------------------===//

/// Random number generation
void fillRandom(float *mat, int64_t numElements);

void fillConstant(float *mat, int64_t numElements, float constant);

void fillLinspace(float *mat, float start, int64_t size, float step);

void printMatrix(const float *mat, int m, int n);

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

#endif // HELLOCUDA_SRC_UTILS_UTILS_H