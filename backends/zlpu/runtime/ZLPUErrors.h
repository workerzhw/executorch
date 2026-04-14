/*
 * Copyright (c) 2026, ZLPU Authors.
 */

#pragma once

#include <cstdint>
#include <string>

namespace executorch {
namespace backends {
namespace zlpu {

enum class Error : uint32_t {
  Ok = 0,
  NotInitialized = 1,
  AlreadyInitialized = 2,
  InvalidPointer = 3,
  InvalidSize = 4,
  InvalidMemorySpace = 5,
  OutOfMemory = 6,
  DriverNotReady = 7,
  DmaFailed = 8,
  CommandSubmitFailed = 9,
  ExecutionFailed = 10,
  Timeout = 11,
  InvalidBinaryFormat = 12,
  TensorNotFound = 13,
  KernelNotFound = 14,
  Unknown = 100,
};

inline const char* error_to_string(Error e) {
  switch (e) {
    case Error::Ok:
      return "Success";
    case Error::NotInitialized:
      return "Not initialized";
    case Error::AlreadyInitialized:
      return "Already initialized";
    case Error::InvalidPointer:
      return "Invalid pointer";
    case Error::InvalidSize:
      return "Invalid size";
    case Error::InvalidMemorySpace:
      return "Invalid memory space";
    case Error::OutOfMemory:
      return "Out of memory";
    case Error::DriverNotReady:
      return "Driver not ready";
    case Error::DmaFailed:
      return "DMA failed";
    case Error::CommandSubmitFailed:
      return "Command submit failed";
    case Error::ExecutionFailed:
      return "Execution failed";
    case Error::Timeout:
      return "Timeout";
    case Error::InvalidBinaryFormat:
      return "Invalid binary format";
    case Error::TensorNotFound:
      return "Tensor not found";
    case Error::KernelNotFound:
      return "Kernel not found";
    default:
      return "Unknown error";
  }
}

} // namespace zlpu
} // namespace backends
} // namespace executorch