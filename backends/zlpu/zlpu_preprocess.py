# Copyright (c) 2026 ZLPU Authors

import copy
from typing import List

import torch
from executorch.backends.zlpu.operators.node_visitor import ZLPUNodeVisitor

from executorch.backends.zlpu.serialization.zlpu_graph_schema import (
    ZLPUComputeOperation,
    ZLPUGraph,
    ZLPUKernelType,
    ZLPUMemorySpace,
    ZLPUTensorValue,
)
from executorch.backends.zlpu.serialization.zlpu_graph_serialize import (
    serialize_zlpu_graph,
)
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


class ZlpuPreprocessor:
    """Preprocessor for ZLPU backend."""

    def __init__(self, edge_program: ExportedProgram):
        self.edge_program = copy.deepcopy(edge_program)

    def preprocess(self) -> PreprocessResult:
        """Convert edge program to ZLPU binary format."""
        graph = self.extract_graph()
        binary_data = serialize_zlpu_graph(graph)
        return PreprocessResult(processed_bytes=binary_data)

    def extract_graph(self) -> ZLPUGraph:
        """Extract ZLPUGraph from edge program."""
        visitor = ZLPUNodeVisitor()
        tensor_values = []

        for node in self.edge_program.graph.nodes:
            if node.op == "placeholder":
                for arg in node.args:
                    if isinstance(arg, torch.Tensor):
                        tensor_val = visitor.get_tensor_value(arg)
                        tensor_values.append(tensor_val)
            elif node.op == "call_function":
                visitor.visit(node)
                for arg in node.args:
                    if isinstance(arg, torch.Tensor):
                        tensor_val = visitor.get_tensor_value(arg)
                        tensor_values.append(tensor_val)
            elif node.op == "call_function":
                for arg in node.args:
                    if isinstance(arg, torch.Tensor):
                        tensor_val = visitor.get_tensor_value(arg)
                        if tensor_val not in tensor_values:
                            tensor_values.append(tensor_val)

        # Get input/output IDs
        input_ids = []
        output_ids = []

        for node in self.edge_program.graph.nodes:
            if node.op == "placeholder":
                for arg in node.args:
                    if isinstance(arg, torch.Tensor):
                        input_ids.append(visitor.get_tensor_id(arg))
            elif node.op == "output":
                for arg in node.args:
                    if isinstance(arg, torch.Tensor):
                        output_ids.append(visitor.get_tensor_id(arg))

        return ZLPUGraph(
            tensor_values=tensor_values,
            operations=visitor.operations,
            input_ids=input_ids,
            output_ids=output_ids,
        )


class ZlpuBackend(BackendDetails):
    """ZLPU Backend details for ExecuTorch."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List,
    ) -> PreprocessResult:
        preprocessor = ZlpuPreprocessor(copy.deepcopy(edge_program))
        return preprocessor.preprocess()
