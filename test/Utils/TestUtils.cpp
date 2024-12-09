#include "src/Utils/Utils.cuh"
#include "gtest/gtest.h"

TEST(Fill, Linspace) {
  Host host;
  int64_t size = 10;
  float *a = host.allocatePageLocked<float>(sizeof(float) * size);
  fillLinspace(a, 0.0f, size, 1.0f);
  for (int64_t i = 0; i < size; i++)
    EXPECT_EQ(a[i], static_cast<float>(i));
}

TEST(Fill, Constant) {
  Host host;
  int64_t size = 10;
  float constant = 2.0f;
  float *a = host.allocatePageLocked<float>(sizeof(float) * size);
  fillConstant(a, size, 2.0f);
  for (int64_t i = 0; i < size; i++)
    EXPECT_EQ(a[i], constant);
}

TEST(HostDevice, PageLocked) {
  // Move data from `locA` -> `devA` and `devA` -> `locB`.
  // Assert `locA[i] == locB[i]` for i in range(data)
  Device device;
  device.set(0);
  Host host;
  constexpr int64_t size = 10;
  constexpr int64_t totalBytes = sizeof(float) * size;
  float *locA = host.allocatePageLocked<float>(totalBytes);
  fillLinspace(locA, 1.0f, size, 2.0f);

  float *devA = device.allocate<float>(totalBytes);
  device.copyFromHost((void *)(locA), (void *)(devA), totalBytes);

  float *locB = host.allocatePageLocked<float>(totalBytes);
  fillConstant(locB, size, 0.0f);

  device.copyToHost((void *)(locB), (void *)(devA), totalBytes);

  for (int64_t i = 0; i < size; i++)
    EXPECT_EQ(locA[i], locB[i]);
}

TEST(HostDevice, ZeroCopy) {
  // Move data from `locA` -> `devA` and `devA` -> `locB`.
  // Assert `locA[i] == locB[i]` for i in range(data)
  Device device;
  device.set(0);
  Host host;
  constexpr int64_t size = 10;
  constexpr int64_t totalBytes = sizeof(float) * size;
  float *locA = host.allocateZeroCopy<float>(totalBytes);
  fillLinspace(locA, 1.0f, size, 2.0f);

  float *devA = device.allocate<float>(totalBytes);
  device.copyFromHost((void *)(locA), (void *)(devA), totalBytes);

  float *locB = host.allocatePageLocked<float>(totalBytes);
  fillConstant(locB, size, 0.0f);

  device.copyToHost((void *)(locB), (void *)(devA), totalBytes);

  for (int64_t i = 0; i < size; i++)
    EXPECT_EQ(locA[i], locB[i]);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}