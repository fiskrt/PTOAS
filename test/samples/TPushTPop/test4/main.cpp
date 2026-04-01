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
#include <cstring>
#include <vector>

void LaunchScope3Incore4Incore0TPushTPopAdd(uint8_t *out, uint8_t *lhs,
                                            uint8_t *rhs, void *stream);

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    aclError _ret = (expr);                                                    \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ACL ERROR] %s failed: %d (%s:%d)\n", #expr,       \
                   (int)_ret, __FILE__, __LINE__);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static uint16_t floatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t roundBias = 0x7FFFu + ((bits >> 16) & 1u);
  bits += roundBias;
  return static_cast<uint16_t>(bits >> 16);
}

static float bf16BitsToFloat(uint16_t bits) {
  const uint32_t storage = static_cast<uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &storage, sizeof(value));
  return value;
}

// Use deterministic small integers so the input is less trivial while BF16
// conversion stays exact.
static float lhsValue(int row, int col) {
  return static_cast<float>(((row * 17 + col * 29 + 3) % 7) - 3);
}

static float rhsValue(int row, int col) {
  return static_cast<float>(((row * 13 + col * 19 + 5) % 9) - 4);
}

int main() {
  constexpr int rows = 16;
  constexpr int depth = 64;
  constexpr int cols = 4 * 128;

  constexpr size_t outBytes = static_cast<size_t>(rows) * cols * sizeof(float);
  constexpr size_t lhsBytes = static_cast<size_t>(rows) * depth * sizeof(uint16_t);
  constexpr size_t rhsBytes = static_cast<size_t>(depth) * cols * sizeof(uint16_t);

  std::vector<float> hostOut(rows * cols, 1.0f);
  std::vector<float> hostGolden(rows * cols, 0.0f);
  std::vector<uint16_t> hostLhs(rows * depth, 0);
  std::vector<uint16_t> hostRhs(depth * cols, 0);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < depth; ++col)
      hostLhs[row * depth + col] = floatToBf16Bits(lhsValue(row, col));
  }

  for (int row = 0; row < depth; ++row) {
    for (int col = 0; col < cols; ++col) {
      const float value = rhsValue(row, col);
      hostRhs[row * cols + col] = floatToBf16Bits(value);
    }
  }

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      float expected = hostOut[row * cols + col];
      for (int k = 0; k < depth; ++k) {
        const float lhs = bf16BitsToFloat(hostLhs[row * depth + k]);
        const float rhs = bf16BitsToFloat(hostRhs[k * cols + col]);
        expected += lhs * rhs;
      }
      hostGolden[row * cols + col] = expected;
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

  LaunchScope3Incore4Incore0TPushTPopAdd(devOut, devLhs, devRhs, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(hostOut.data(), outBytes, devOut, outBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t idx = static_cast<size_t>(row) * cols + col;
      if (std::fabs(hostOut[idx] - hostGolden[idx]) > 1e-5f) {
        std::fprintf(stderr,
                     "Mismatch at (%d, %d): got %.6f, expect %.6f\n",
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

  std::puts("scope3_incore_4_incore_0_tpush_tpop_add passed. "
            "Validated one 16x512 output window.");

  aclrtFree(devRhs);
  aclrtFree(devLhs);
  aclrtFree(devOut);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
