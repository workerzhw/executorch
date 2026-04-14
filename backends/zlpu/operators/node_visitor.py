# Copyright (c) 2026 ZLPU Authors

import torch
from executorch.backends.zlpu.serialization.zlpu_graph_schema import (
    ZLPUComputeOperation,
    ZLPUKernelType,
    ZLPUMemorySpace,
    ZLPUTensorValue,
)


class ZLPUNodeVisitor:
    """Visitor to convert ExecuTorch graph nodes to ZLPU operations."""

    KERNEL_MAP = {
        "aten.linear": ZLPUKernelType.LINEAR,
        "aten.matmul": ZLPUKernelType.MATMUL,
        "aten.addmm": ZLPUKernelType.ADDMM,
        "aten.layer_norm": ZLPUKernelType.LAYER_NORM,
        "aten.rms_norm": ZLPUKernelType.RMS_NORM,
        "aten.softmax": ZLPUKernelType.SOFTMAX,
        "aten.silu": ZLPUKernelType.SILU,
        "aten.gelu": ZLPUKernelType.GELU,
        "aten.sigmoid": ZLPUKernelType.SIGMOID,
        "aten.relu": ZLPUKernelType.RELU,
        "aten.cat": ZLPUKernelType.CAT,
        "aten.view": ZLPUKernelType.VIEW,
        "aten.reshape": ZLPUKernelType.RESHAPE,
        "aten.scaled_dot_product_attention": ZLPUKernelType.SDPA,
    }

    def __init__(self):
        self.tensor_id_map = {}
        self.next_tensor_id = 0
        self.operations = []

    def get_tensor_id(self, val) -> int:
        """Get or assign tensor ID."""
        if id(val) not in self.tensor_id_map:
            self.tensor_id_map[id(val)] = self.next_tensor_id
            self.next_tensor_id += 1
        return self.tensor_id_map[id(val)]

    def get_tensor_value(self, val) -> ZLPUTensorValue:
        """Convert a value to ZLPUTensorValue."""
        tensor_id = self.get_tensor_id(val)
        if isinstance(val, Tensor):
            return ZLPUTensorValue(
                id=tensor_id,
                shape=list(val.shape),
                dtype=val.dtype,
                memory_space=ZLPUMemorySpace.DDR,
            )
        return ZLPUTensorValue(
            id=tensor_id,
            shape=[],
            dtype=torch.float32,
            memory_space=ZLPUMemorySpace.DDR,
        )

    def visit(self, node):
        """Visit a node and convert to ZLPU operation."""
        if node.op == "call_function":
            self.visit_call_function(node)

    def visit_call_function(self, node):
        """Handle call_function nodes."""
        if node.target.__name__ in self.KERNEL_MAP:
            self.visit_compute_node(node)

    def visit_compute_node(self, node):
        """Convert compute node to ZLPUComputeOperation."""
        kernel_type = self.KERNEL_MAP[node.target.__name__]

        input_ids = []
        weight_ids = []
        output_id = self.get_tensor_id(node.args[0])

        # First arg is usually output for most ops
        if len(node.args) >= 2:
            input_ids.append(self.get_tensor_id(node.args[1]))
        if len(node.args) >= 3:
            weight_ids.append(self.get_tensor_id(node.args[2]))

        for arg in node.args[1:]:
            if isinstance(arg, Tensor):
                if len(weight_ids) > 0:
                    weight_ids.append(self.get_tensor_id(arg))
                else:
                    input_ids.append(self.get_tensor_id(arg))

        op = ZLPUComputeOperation(
            kernel_type=kernel_type,
            input_ids=input_ids,
            weight_ids=weight_ids,
            output_id=output_id,
        )
        self.operations.append(op)
