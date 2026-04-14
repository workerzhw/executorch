/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#include "ZLPUDriverStub.h"
#include <cstring>
#include <iostream>
#include "ZLPUScpCommand.h"

namespace executorch {
namespace backends {
namespace zlpu {

ZLPUDriverStub::ZLPUDriverStub()
    : l1m_base_(nullptr),
      r1m_base_(nullptr),
      l1m_used_(0),
      r1m_used_(0),
      initialized_(false) {}

ZLPUDriverStub::~ZLPUDriverStub() {
  if (l1m_base_) {
    free(l1m_base_);
  }
  if (r1m_base_) {
    free(r1m_base_);
  }
}

Error ZLPUDriverStub::initialize() {
  if (initialized_) {
    return Error::AlreadyInitialized;
  }

  l1m_base_ = malloc(kL1MSize);
  r1m_base_ = malloc(kR1MSize);

  if (!l1m_base_ || !r1m_base_) {
    return Error::OutOfMemory;
  }

  memset(l1m_base_, 0, kL1MSize);
  memset(r1m_base_, 0, kR1MSize);

  l1m_used_ = 0;
  r1m_used_ = 0;
  initialized_ = true;

  return Error::Ok;
}

Error ZLPUDriverStub::shutdown() {
  if (!initialized_) {
    return Error::NotInitialized;
  }

  if (l1m_base_) {
    free(l1m_base_);
    l1m_base_ = nullptr;
  }
  if (r1m_base_) {
    free(r1m_base_);
    r1m_base_ = nullptr;
  }

  initialized_ = false;
  return Error::Ok;
}

Error ZLPUDriverStub::allocateL1M(void** ptr, size_t size) {
  if (!initialized_) {
    return Error::NotInitialized;
  }
  if (!ptr || size == 0) {
    return Error::InvalidPointer;
  }
  if (l1m_used_ + size > kL1MSize) {
    return Error::OutOfMemory;
  }

  *ptr = static_cast<char*>(l1m_base_) + l1m_used_;
  l1m_used_ += size;
  return Error::Ok;
}

Error ZLPUDriverStub::freeL1M(void* ptr) {
  (void)ptr;
  return Error::Ok;
}

Error ZLPUDriverStub::allocateR1M(void** ptr, size_t size) {
  if (!initialized_) {
    return Error::NotInitialized;
  }
  if (!ptr || size == 0) {
    return Error::InvalidPointer;
  }
  if (r1m_used_ + size > kR1MSize) {
    return Error::OutOfMemory;
  }

  *ptr = static_cast<char*>(r1m_base_) + r1m_used_;
  r1m_used_ += size;
  return Error::Ok;
}

Error ZLPUDriverStub::freeR1M(void* ptr) {
  (void)ptr;
  return Error::Ok;
}

Error ZLPUDriverStub::dmaCopy(
    void* dst,
    const void* src,
    size_t size,
    MemorySpace dst_space,
    MemorySpace src_space) {
  if (!initialized_) {
    return Error::NotInitialized;
  }
  if (!dst || !src) {
    return Error::InvalidPointer;
  }

  // For stub, just do memcpy
  (void)dst_space;
  (void)src_space;
  memcpy(dst, src, size);
  return Error::Ok;
}

Error ZLPUDriverStub::submitCommand(const ScpCommandHeader& cmd) {
  (void)cmd;
  return Error::Ok;
}

Error ZLPUDriverStub::waitForCompletion(uint32_t timeout_ms) {
  (void)timeout_ms;
  return Error::Ok;
}

bool ZLPUDriverStub::isReady() const {
  return initialized_;
}

std::unique_ptr<ZLPUDriverInterface> createZLPUDriver() {
  return std::make_unique<ZLPUDriverStub>();
}

} // namespace zlpu
} // namespace backends
} // namespace executorch