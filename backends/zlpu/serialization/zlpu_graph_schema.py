# Copyright (c) 2026 ZLPU Authors

from dataclasses import dataclass
from enum import IntEnum
from typing import List

import torch


class ZLPUMemorySpace(IntEnum):
    DDR = 0
    L1M = 1
    R1M = 2


class ZLPUKernelType(IntEnum):
    LINEAR = 1
    MATMUL = 2
    ADDMM = 3
    LAYER_NORM = 4
    RMS_NORM = 5
    SOFTMAX = 6
    SILU = 7
    GELU = 8
    SIGMOID = 9
    RELU = 10
    CAT = 11
    VIEW = 12
    RESHAPE = 13
    SDPA = 14


@dataclass
class ZLPUTensorValue:
    id: int
    shape: List[int]
    dtype: torch.dtype
    memory_space: ZLPUMemorySpace = ZLPUMemorySpace.DDR
    offset: int = 0


@dataclass
class ZLPUDmaOperation:
    src_addr: int
    dst_addr: int
    transfer_size: int
    src_space: ZLPUMemorySpace
    dst_space: ZLPUMemorySpace


@dataclass
class ZLPUComputeOperation:
    kernel_type: ZLPUKernelType
    input_ids: List[int]
    weight_ids: List[int]
    output_id: int


@dataclass
class ZLPUGraph:
    tensor_values: List[ZLPUTensorValue]
    operations: List
    input_ids: List[int]
    output_ids: List[int]
