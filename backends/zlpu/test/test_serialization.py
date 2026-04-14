# Copyright (c) 2026 ZLPU Authors
"""
Test serialization for ZLPU backend.
"""

import os
import struct
import sys

import torch

# Add the project root to path
project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, project_root)

try:
    from executorch.backends.zlpu.serialization.zlpu_graph_schema import (
        ZLPUComputeOperation,
        ZLPUDmaOperation,
        ZLPUGraph,
        ZLPUKernelType,
        ZLPUMemorySpace,
        ZLPUTensorValue,
    )
    from executorch.backends.zlpu.serialization.zlpu_graph_serialize import (
        MAGIC,
        serialize_zlpu_graph,
    )
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Note: executorch package not installed. Running standalone test...")
    # Run standalone test without executorch imports
    sys.path.insert(0, os.path.join(project_root, "backends", "zlpu"))
    from serialization.zlpu_graph_schema import (
        ZLPUComputeOperation,
        ZLPUDmaOperation,
        ZLPUGraph,
        ZLPUKernelType,
        ZLPUMemorySpace,
        ZLPUTensorValue,
    )
    from serialization.zlpu_graph_serialize import MAGIC, serialize_zlpu_graph


def test_serialization():
    """Test serialization of ZLPU graph."""
    tensor_values = [
        ZLPUTensorValue(
            id=0, shape=[1, 128], dtype=torch.float32, memory_space=ZLPUMemorySpace.DDR
        ),
        ZLPUTensorValue(
            id=1, shape=[128, 64], dtype=torch.float32, memory_space=ZLPUMemorySpace.R1M
        ),
        ZLPUTensorValue(
            id=2, shape=[1, 64], dtype=torch.float32, memory_space=ZLPUMemorySpace.DDR
        ),
    ]
    operations = [
        ZLPUComputeOperation(
            kernel_type=ZLPUKernelType.LINEAR,
            input_ids=[0],
            weight_ids=[1],
            output_id=2,
        ),
    ]
    input_ids = [0]
    output_ids = [2]

    graph = ZLPUGraph(
        tensor_values=tensor_values,
        operations=operations,
        input_ids=input_ids,
        output_ids=output_ids,
    )

    # Serialize
    data = serialize_zlpu_graph(graph)

    # Verify header
    assert data[:4] == MAGIC, f"Invalid MAGIC: {data[:4]}"
    assert len(data) > 20, "Data too short"

    # Verify header fields
    (version,) = struct.unpack("<H", data[4:6])
    assert version == 0x0100, f"Invalid version: {version}"

    print("Serialization test PASSED!")


def test_serialization_with_dma():
    """Test serialization with DMA operations."""
    tensor_values = [
        ZLPUTensorValue(
            id=0, shape=[1, 128], dtype=torch.float32, memory_space=ZLPUMemorySpace.DDR
        ),
        ZLPUTensorValue(
            id=1, shape=[128, 64], dtype=torch.float32, memory_space=ZLPUMemorySpace.R1M
        ),
    ]
    operations = [
        ZLPUDmaOperation(
            src_addr=0,
            dst_addr=1,
            transfer_size=128 * 64 * 4,
            src_space=ZLPUMemorySpace.DDR,
            dst_space=ZLPUMemorySpace.R1M,
        ),
    ]
    input_ids = [0]
    output_ids = [1]

    graph = ZLPUGraph(
        tensor_values=tensor_values,
        operations=operations,
        input_ids=input_ids,
        output_ids=output_ids,
    )

    data = serialize_zlpu_graph(graph)
    assert len(data) > 20
    print("DMA serialization test PASSED!")


if __name__ == "__main__":
    test_serialization()
    test_serialization_with_dma()
