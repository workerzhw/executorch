# Copyright (c) 2026 ZLPU Authors

from typing import List

from executorch.exir.backend.backend_details import ExportedProgram
from executorch.exir.backend.partitioner import Partitioner, PartitionResult


class ZlpuPartitioner(Partitioner):
    """Partitioner for ZLPU backend."""

    SUPPORTED_OPS = [
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

    def __init__(self):
        super().__init__()

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """Partition the graph for ZLPU backend."""
        partition_tags = {}

        for node in exported_program.graph.nodes:
            if node.op == "call_function":
                if hasattr(node, "target") and hasattr(node.target, "__name__"):
                    op_name = node.target.__name__
                    if op_name in self.SUPPORTED_OPS:
                        partition_tags[node.name] = "zlpu"

        return PartitionResult(
            partition_tags=partition_tags,
            # additional_copy_nodes: not implemented
        )

    def get_supported_ops(self) -> List[str]:
        """Return list of supported operators."""
        return self.SUPPORTED_OPS
