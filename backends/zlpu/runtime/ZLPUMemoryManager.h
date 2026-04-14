/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include "ZLPUDriverInterface.h"
#include "ZLPUScpCommand.h"

namespace executorch {
namespace backends {
namespace zlpu {

struct TensorInfo {
  uint32_t id;
  std::vector<int32_t> shape;
  MemorySpace memory_space;
  size_t size;
  void* data_ptr;
};

class ZLPUMemoryManager {
 public:
  ZLPUMemoryManager(ZLPUDriverInterface* driver);
  ~ZLPUMemoryManager();

  Error allocateTensor(
      uint32_t id,
      const std::vector<int32_t>& shape,
      MemorySpace space);
  Error freeTensor(uint32_t id);

  void* getTensorPtr(uint32_t id) const;
  TensorInfo* getTensorInfo(uint32_t id) const;

 private:
  ZLPUDriverInterface* driver_;
  std::unordered_map<uint32_t, TensorInfo> tensors_;
};

} // namespace zlpu
} // namespace backends
} // namespace executorch