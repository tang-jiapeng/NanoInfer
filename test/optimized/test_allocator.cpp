#include <glog/logging.h>
#include <gtest/gtest.h>
#include "nanoinfer/tensor/tensor.h"
#include "nanoinfer/base/buffer.h"
TEST(test_buffer, use_external1) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  CHECK_EQ(buffer.is_external(), true);
  free(buffer.ptr());
}