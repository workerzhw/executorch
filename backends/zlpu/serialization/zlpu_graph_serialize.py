# Copyright (c) 2026 ZLPU Authors

import struct
from typing import List

import torch

from .zlpu_graph_schema import (
    ZLPUComputeOperation,
    ZLPUDmaOperation,
    ZLPUGraph,
    ZLPUKernelType,
    ZLPUMemorySpace,
    ZLPUTensorValue,
)

MAGIC = b"ZLPU"
VERSION = 0x0100


def serialize_zlpu_graph(graph: ZLPUGraph) -> bytes:
    """Serialize ZLPUGraph to binary format."""
    data = bytearray()

    # Header: magic(4) + version(2) + num_ops(4) + num_tensors(4) + constant_data_size(4)
    num_ops = len(graph.operations)
    num_tensors = len(graph.tensor_values)
    constant_data_size = 0

    data.extend(MAGIC)
    data.extend(struct.pack("<H", VERSION))
    data.extend(struct.pack("<I", num_ops))
    data.extend(struct.pack("<I", num_tensors))
    data.extend(struct.pack("<I", constant_data_size))

    # Serialize operations
    for op in graph.operations:
        if isinstance(op, ZLPUComputeOperation):
            data.extend(serialize_compute_op(op))
        elif isinstance(op, ZLPUDmaOperation):
            data.extend(serialize_dma_op(op))

    # Serialize tensors
    for tensor in graph.tensor_values:
        data.extend(serialize_tensor(tensor))

    # Constant data (empty for now)
    data.extend(b"\x00" * constant_data_size)

    return bytes(data)


def serialize_compute_op(op: ZLPUComputeOperation) -> bytes:
    """Serialize compute operation."""
    data = bytearray()

    # OpHeader
    op_type = 0x02  # COMPUTE
    flags = 0
    payload_size = 16 + len(op.input_ids) * 4 + len(op.weight_ids) * 4
    size = 8 + payload_size  # header(8) + payload

    data.extend(struct.pack("<H", op_type))
    data.extend(struct.pack("<H", flags))
    data.extend(struct.pack("<I", size))
    data.extend(struct.pack("<Q", 0))  # reserved

    # Payload
    data.extend(struct.pack("<I", op.kernel_type))
    data.extend(struct.pack("<I", len(op.input_ids)))
    for inp_id in op.input_ids:
        data.extend(struct.pack("<I", inp_id))
    data.extend(struct.pack("<I", len(op.weight_ids)))
    for w_id in op.weight_ids:
        data.extend(struct.pack("<I", w_id))
    data.extend(struct.pack("<I", op.output_id))
    data.extend(struct.pack("<I", 0))  # param_addr

    return bytes(data)


def serialize_dma_op(op: ZLPUDmaOperation) -> bytes:
    """Serialize DMA operation."""
    data = bytearray()

    # OpHeader
    op_type = 0x01  # DMA
    flags = 0
    size = 8 + 24  # header(8) + payload(24)

    data.extend(struct.pack("<H", op_type))
    data.extend(struct.pack("<H", flags))
    data.extend(struct.pack("<I", size))
    data.extend(struct.pack("<Q", 0))  # reserved

    # Payload
    data.extend(struct.pack("<I", op.src_addr))
    data.extend(struct.pack("<I", op.dst_addr))
    data.extend(struct.pack("<I", op.transfer_size))
    data.extend(struct.pack("<B", op.src_space))
    data.extend(struct.pack("<B", op.dst_space))
    data.extend(struct.pack("<H", 0))  # reserved

    return bytes(data)


def serialize_tensor(tensor: ZLPUTensorValue) -> bytes:
    """Serialize tensor metadata."""
    data = bytearray()

    data.extend(struct.pack("<I", tensor.id))
    data.extend(struct.pack("<I", len(tensor.shape)))
    for dim in tensor.shape:
        data.extend(struct.pack("<I", dim))

    # dtype mapping
    dtype_map = {torch.float32: 6, torch.float16: 5, torch.int32: 3}
    dtype_val = dtype_map.get(tensor.dtype, 6)
    data.extend(struct.pack("<I", dtype_val))

    data.extend(struct.pack("<B", tensor.memory_space))
    data.extend(struct.pack("<Q", tensor.offset))

    return bytes(data)
