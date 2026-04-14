/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#include "ZLPUMemoryManager.h"
#include <cstring>
#include "ZLPUDriverInterface.h"

namespace executorch {
namespace backends {
namespace zlpu {

ZLPUMemoryManager::ZLPUMemoryManager(ZLPUDriverInterface* driver)
    : driver_(driver) {}

ZLPUMemoryManager::~ZLPUMemoryManager() {
  for (auto& [id, info] : tensors_) {
    if (info.data_ptr) {
      if (info.memory_space == MemorySpace::L1M) {
        driver_->freeL1M(info.data_ptr);
      } else if (info.memory_space == MemorySpace::R1M) {
        driver_->freeR1M(info.data_ptr);
      }
    }
  }
}

Error ZLPUMemoryManager::allocateTensor(
    uint32_t id,
    const std::vector<int32_t>& shape,
    MemorySpace space) {
  // Calculate size
  size_t size = 4; // float32
  for (int32_t dim : shape) {
    size *= dim;
  }

  void* ptr = nullptr;
  if (space == MemorySpace::L1M) {
    auto err = driver_->allocateL1M(&ptr, size);
    if (err != Error::Ok) {
      return err;
    }
  } else if (space == MemorySpace::R1M) {
    auto err = driver_->allocateR1M(&ptr, size);
    if (err != Error::Ok) {
      return err;
    }
  } else {
    ptr = malloc(size);
    if (!ptr) {
      return Error::OutOfMemory;
    }
  }

  TensorInfo info;
  info.id = id;
  info.shape = shape;
  info.memory_space = space;
  info.size = size;
  info.data_ptr = ptr;

  tensors_[id] = info;
  return Error::Ok;
}

Error ZLPUMemoryManager::freeTensor(uint32_t id) {
  auto it = tensors_.find(id);
  if (it == tensors_.end()) {
    return Error::TensorNotFound;
  }

  TensorInfo& info = it->second;
  if (info.data_ptr) {
    if (info.memory_space == MemorySpace::L1M) {
      driver_->freeL1M(info.data_ptr);
    } else if (info.memory_space == MemorySpace::R1M) {
      driver_->freeR1M(info.data_ptr);
    } else {
      free(info.data_ptr);
    }
  }

  tensors_.erase(it);
  return Error::Ok;
}

void* ZLPUMemoryManager::getTensorPtr(uint32_t id) const {
  auto it = tensors_.find(id);
  if (it != tensors_.end()) {
    return it->second.data_ptr;
  }
  return nullptr;
}

TensorInfo* ZLPUMemoryManager::getTensorInfo(uint32_t id) const {
  auto it = tensors_.find(id);
  if (it != tensors_.end()) {
    return const_cast<TensorInfo*>(&it->second);
  }
  return nullptr;
}

} // namespace zlpu
} // namespace backends
} // namespace executorch