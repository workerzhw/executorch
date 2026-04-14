/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#pragma once

#include <memory>
#include "ZLPUErrors.h"
#include "ZLPUScpCommand.h"

namespace executorch {
namespace backends {
namespace zlpu {

class ZLPUDriverInterface {
 public:
  virtual ~ZLPUDriverInterface() = default;

  virtual Error initialize() = 0;
  virtual Error shutdown() = 0;

  virtual Error allocateL1M(void** ptr, size_t size) = 0;
  virtual Error freeL1M(void* ptr) = 0;

  virtual Error allocateR1M(void** ptr, size_t size) = 0;
  virtual Error freeR1M(void* ptr) = 0;

  virtual Error dmaCopy(
      void* dst,
      const void* src,
      size_t size,
      MemorySpace dst_space,
      MemorySpace src_space) = 0;

  virtual Error submitCommand(const ScpCommandHeader& cmd) = 0;
  virtual Error waitForCompletion(uint32_t timeout_ms) = 0;

  virtual bool isReady() const = 0;
};

std::unique_ptr<ZLPUDriverInterface> createZLPUDriver();

} // namespace zlpu
} // namespace backends
} // namespace executorch