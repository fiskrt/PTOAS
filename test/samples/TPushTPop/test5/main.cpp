// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

void LaunchTpushTpopLeftRightAdd(uint8_t *out, uint8_t *lhs, uint8_t *rhs,
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

static float lhsValue(int row, int col) {
  return static_cast<float>(((row * 17 + col * 29 + 3) % 11) - 5);
}

static float rhsValue(int row, int col) {
  return static_cast<float>(((row * 13 + col * 19 + 7) % 13) - 6);
}

int main() {
  constexpr int M = 16;
  constexpr int K = 64;
  constexpr int N = 128;

  constexpr size_t outBytes = static_cast<size_t>(M) * N * sizeof(float);
  constexpr size_t lhsBytes = static_cast<size_t>(M) * K * sizeof(float);
  constexpr size_t rhsBytes = static_cast<size_t>(K) * N * sizeof(float);

  std::vector<float> hostOut(M * N, 1.0f);
  std::vector<float> hostGolden(M * N, 1.0f);
  std::vector<float> hostLhs(M * K, 0.0f);
  std::vector<float> hostRhs(K * N, 0.0f);

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < K; ++col)
      hostLhs[row * K + col] = lhsValue(row, col);
  }

  for (int row = 0; row < K; ++row) {
    for (int col = 0; col < N; ++col)
      hostRhs[row * N + col] = rhsValue(row, col);
  }

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float expected = hostOut[row * N + col];
      for (int k = 0; k < K; ++k)
        expected += hostLhs[row * K + k] * hostRhs[k * N + col];
      hostGolden[row * N + col] = expected;
    }
  }

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  uint8_t *devOut = nullptr;
  uint8_t *devLhs = nullptr;
  uint8_t *devRhs = nullptr;
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devOut), outBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devLhs), lhsBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devRhs), rhsBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACL_CHECK(aclrtMemcpy(devOut, outBytes, hostOut.data(), outBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devLhs, lhsBytes, hostLhs.data(), lhsBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devRhs, rhsBytes, hostRhs.data(), rhsBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchTpushTpopLeftRightAdd(devOut, devLhs, devRhs, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(hostOut.data(), outBytes, devOut, outBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      const size_t idx = static_cast<size_t>(row) * N + col;
      if (std::fabs(hostOut[idx] - hostGolden[idx]) > 1e-5f) {
        std::fprintf(stderr, "Mismatch at (%d, %d): got %.6f, expect %.6f\n",
                     row, col, hostOut[idx], hostGolden[idx]);
        aclrtFree(devRhs);
        aclrtFree(devLhs);
        aclrtFree(devOut);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
      }
    }
  }

  std::puts("tpush_tpop_left_right_add passed.");

  aclrtFree(devRhs);
  aclrtFree(devLhs);
  aclrtFree(devOut);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
