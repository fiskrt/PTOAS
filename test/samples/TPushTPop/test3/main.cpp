// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

void LaunchRoundTripTPushTPopPrint(uint8_t *src, uint8_t *identity,
                                   void *stream);

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    aclError _ret = (expr);                                                    \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ACL ERROR] %s failed: %d (%s:%d)\n", #expr,       \
                   (int)_ret, __FILE__, __LINE__);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  constexpr int rows = 16;
  constexpr int cols = 16;
  constexpr size_t srcBytes = rows * cols * sizeof(float);
  constexpr size_t identityBytes = rows * cols * sizeof(float);

  std::vector<float> hostSrc(rows * cols, 0.0f);
  std::vector<float> hostIdentity(rows * cols, 0.0f);
  for (int i = 0; i < rows * cols; ++i)
    hostSrc[i] = static_cast<float>(i);
  for (int i = 0; i < rows; ++i)
    hostIdentity[i * cols + i] = 1.0f;

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  uint8_t *devSrc = nullptr;
  uint8_t *devIdentity = nullptr;
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devSrc), srcBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devIdentity), identityBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(devSrc, srcBytes, hostSrc.data(), srcBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devIdentity, identityBytes, hostIdentity.data(),
                        identityBytes, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchRoundTripTPushTPopPrint(devSrc, devIdentity, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::puts("Kernel finished. Expect one 16x16 TPRINT block with row-major "
            "values from 0 to 255 after vector->cube->vector round-trip.");

  aclrtFree(devIdentity);
  aclrtFree(devSrc);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
