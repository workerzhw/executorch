/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace executorch {
namespace backends {
namespace zlpu {

constexpr uint32_t kMagic = 0x5A4C5055; // "ZLPU"
constexpr uint16_t kVersion = 0x0100;
constexpr size_t kL1MSize = 2 * 1024 * 1024; // 2MB
constexpr size_t kR1MSize = 3 * 1024 * 1024; // 3MB

enum class MemorySpace : uint8_t {
  DDR = 0,
  L1M = 1,
  R1M = 2,
};

enum class OpType : uint16_t {
  DMA = 0x01,
  COMPUTE = 0x02,
  SYNC = 0x03,
};

enum class KernelType : uint32_t {
  LINEAR = 1,
  MATMUL = 2,
  ADDMM = 3,
  LAYER_NORM = 4,
  RMS_NORM = 5,
  SOFTMAX = 6,
  SILU = 7,
  GELU = 8,
  SIGMOID = 9,
  RELU = 10,
  CAT = 11,
  VIEW = 12,
  RESHAPE = 13,
  SDPA = 14,
};

struct [[gnu::packed]] ScpCommandHeader {
  uint16_t type;
  uint16_t flags;
  uint32_t size;
  uint64_t reserved;
};

struct [[gnu::packed]] ScpDmaCommand {
  ScpCommandHeader header;
  uint32_t src_addr;
  uint32_t dst_addr;
  uint32_t transfer_size;
  uint8_t src_space;
  uint8_t dst_space;
  uint16_t reserved;
};

struct [[gnu::packed]] ScpComputeCommand {
  ScpCommandHeader header;
  uint32_t kernel_id;
  uint32_t num_inputs;
  uint32_t input_ids[8];
  uint32_t num_weights;
  uint32_t weight_ids[8];
  uint32_t output_id;
  uint32_t param_addr;
};

struct [[gnu::packed]] ScpSyncCommand {
  ScpCommandHeader header;
  uint32_t wait_event_id;
};

} // namespace zlpu
} // namespace backends
} // namespace executorch