/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "ZLPUDriverInterface.h"
#include "ZLPUErrors.h"
#include "ZLPUMemoryManager.h"
#include "ZLPUScpCommand.h"

namespace executorch {
namespace backends {
namespace zlpu {

struct BinaryHeader {
  uint32_t magic;
  uint16_t version;
  uint32_t num_ops;
  uint32_t num_tensors;
  uint32_t constant_data_size;
};

struct DmaOperation {
  uint32_t src_addr;
  uint32_t dst_addr;
  uint32_t transfer_size;
  MemorySpace src_space;
  MemorySpace dst_space;
};

struct ComputeOperation {
  KernelType kernel_type;
  std::vector<uint32_t> input_ids;
  std::vector<uint32_t> weight_ids;
  uint32_t output_id;
};

struct ProgramData {
  BinaryHeader header;
  std::vector<DmaOperation> dma_ops;
  std::vector<ComputeOperation> compute_ops;
  std::vector<uint8_t> constant_data;
  size_t current_offset;
};

class ZLPUExecutor {
 public:
  ZLPUExecutor(ZLPUDriverInterface* driver, void* l1m_base, void* r1m_base);
  ~ZLPUExecutor();

  Error loadProgram(const void* data, size_t size);
  Error forward();

 private:
  Error parseProgram(const uint8_t* data, size_t size);
  Error executeDma(const DmaOperation& op);
  Error executeCompute(const ComputeOperation& op);

  ZLPUDriverInterface* driver_;
  std::unique_ptr<ZLPUMemoryManager> memory_manager_;
  ProgramData program_;
};

class ZLPUSBackend {
 public:
  ZLPUSBackend();
  ~ZLPUSBackend();

  bool isAvailable() const;
  Error init(void* data, size_t size);
  Error execute();
  void destroy();

 private:
  std::unique_ptr<ZLPUDriverInterface> driver_;
  std::unique_ptr<ZLPUExecutor> executor_;
  bool initialized_;
};

} // namespace zlpu
} // namespace backends
} // namespace executorch