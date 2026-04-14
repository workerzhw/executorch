# Copyright (c) 2026 ZLPU Authors
"""
Test partitioner for ZLPU backend.
"""

import torch
from executorch.backends.zlpu.partition.zlpu_partitioner import ZlpuPartitioner


def test_partitioner():
    """Test partitioner supported ops."""
    expected = [
        "aten.linear",
        "aten.matmul",
        "aten.addmm",
        "aten.layer_norm",
        "aten.rms_norm",
        "aten.softmax",
        "aten.silu",
        "aten.gelu",
        "aten.sigmoid",
        "aten.relu",
        "aten.cat",
        "aten.view",
        "aten.reshape",
        "aten.scaled_dot_product_attention",
    ]

    for op in expected:
        assert op in ZlpuPartitioner.SUPPORTED_OPS, f"{op} not supported"

    print(f"Partitioner supports {len(ZlpuPartitioner.SUPPORTED_OPS)} operators")
    print("Partitioner test PASSED!")


if __name__ == "__main__":
    test_partitioner()
