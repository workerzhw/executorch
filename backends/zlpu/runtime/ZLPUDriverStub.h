/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#pragma once

#include "ZLPUDriverInterface.h"

namespace executorch {
namespace backends {
namespace zlpu {

class ZLPUDriverStub : public ZLPUDriverInterface {
 public:
  ZLPUDriverStub();
  ~ZLPUDriverStub() override;

  Error initialize() override;
  Error shutdown() override;

  Error allocateL1M(void** ptr, size_t size) override;
  Error freeL1M(void* ptr) override;

  Error allocateR1M(void** ptr, size_t size) override;
  Error freeR1M(void* ptr) override;

  Error dmaCopy(
      void* dst,
      const void* src,
      size_t size,
      MemorySpace dst_space,
      MemorySpace src_space) override;

  Error submitCommand(const ScpCommandHeader& cmd) override;
  Error waitForCompletion(uint32_t timeout_ms) override;

  bool isReady() const override;

 private:
  void* l1m_base_;
  void* r1m_base_;
  size_t l1m_used_;
  size_t r1m_used_;
  bool initialized_;
};

} // namespace zlpu
} // namespace backends
} // namespace executorch