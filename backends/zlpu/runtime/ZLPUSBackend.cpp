/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#include "ZLPUSBackend.h"
#include <cstring>
#include <iostream>
#include "ZLPUDriverStub.h"

namespace executorch {
namespace backends {
namespace zlpu {

constexpr uint32_t kExpectedMagic = 0x5A4C5055; // "ZLPU"

ZLPUExecutor::ZLPUExecutor(ZLPUDriverInterface* driver, void*, void*)
    : driver_(driver) {
  memory_manager_ = std::make_unique<ZLPUMemoryManager>(driver_);
}

ZLPUExecutor::~ZLPUExecutor() = default;

Error ZLPUExecutor::loadProgram(const void* data, size_t size) {
  return parseProgram(static_cast<const uint8_t*>(data), size);
}

Error ZLPUExecutor::parseProgram(const uint8_t* data, size_t size) {
  if (size < sizeof(BinaryHeader)) {
    return Error::InvalidBinaryFormat;
  }

  std::memcpy(&program_.header, data, sizeof(BinaryHeader));

  if (program_.header.magic != kExpectedMagic) {
    return Error::InvalidBinaryFormat;
  }

  size_t offset = sizeof(BinaryHeader);
  program_.constant_data.resize(program_.header.constant_data_size);

  if (offset + program_.header.constant_data_size <= size) {
    std::memcpy(
        program_.constant_data.data(),
        data + offset,
        program_.header.constant_data_size);
  }

  return Error::Ok;
}

Error ZLPUExecutor::executeDma(const DmaOperation& op) {
  void* dst = nullptr;
  void* src = nullptr;

  if (op.dst_space == MemorySpace::L1M || op.dst_space == MemorySpace::R1M) {
    dst = memory_manager_->getTensorPtr(op.dst_addr);
  }
  if (op.src_space == MemorySpace::L1M || op.src_space == MemorySpace::R1M) {
    src = memory_manager_->getTensorPtr(op.src_addr);
  }

  return driver_->dmaCopy(
      dst, src, op.transfer_size, op.dst_space, op.src_space);
}

Error ZLPUExecutor::executeCompute(const ComputeOperation& op) {
  ScpComputeCommand cmd;
  std::memset(&cmd, 0, sizeof(cmd));
  cmd.header.type = static_cast<uint16_t>(OpType::COMPUTE);
  cmd.header.size = sizeof(ScpComputeCommand);
  cmd.kernel_id = static_cast<uint32_t>(op.kernel_type);
  cmd.num_inputs = op.input_ids.size();
  for (size_t i = 0; i < op.input_ids.size() && i < 8; ++i) {
    cmd.input_ids[i] = op.input_ids[i];
  }
  cmd.num_weights = op.weight_ids.size();
  for (size_t i = 0; i < op.weight_ids.size() && i < 8; ++i) {
    cmd.weight_ids[i] = op.weight_ids[i];
  }
  cmd.output_id = op.output_id;

  auto err = driver_->submitCommand(cmd.header);
  if (err != Error::Ok) {
    return err;
  }

  return driver_->waitForCompletion(1000);
}

Error ZLPUExecutor::forward() {
  for (const auto& dma : program_.dma_ops) {
    auto err = executeDma(dma);
    if (err != Error::Ok) {
      return err;
    }
  }

  for (const auto& compute : program_.compute_ops) {
    auto err = executeCompute(compute);
    if (err != Error::Ok) {
      return err;
    }
  }

  return Error::Ok;
}

ZLPUSBackend::ZLPUSBackend() : initialized_(false) {
  driver_ = createZLPUDriver();
}

ZLPUSBackend::~ZLPUSBackend() {
  if (initialized_) {
    driver_->shutdown();
  }
}

bool ZLPUSBackend::isAvailable() const {
  return true;
}

Error ZLPUSBackend::init(void* data, size_t size) {
  if (!initialized_) {
    auto err = driver_->initialize();
    if (err != Error::Ok) {
      return err;
    }
    initialized_ = true;
  }

  void* l1m_base = nullptr;
  void* r1m_base = nullptr;
  driver_->allocateL1M(&l1m_base, kL1MSize);
  driver_->allocateR1M(&r1m_base, kR1MSize);

  executor_ = std::make_unique<ZLPUExecutor>(driver_.get(), l1m_base, r1m_base);
  return executor_->loadProgram(data, size);
}

Error ZLPUSBackend::execute() {
  if (!executor_) {
    return Error::NotInitialized;
  }
  return executor_->forward();
}

void ZLPUSBackend::destroy() {
  delete this;
}

} // namespace zlpu
} // namespace backends
} // namespace executorch