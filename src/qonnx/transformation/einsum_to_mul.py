import numpy as np

from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


class EinsumToMul(Transformation):
    """Replace broadcastable two-input Einsum nodes with Mul nodes.

    The transformation supports equations without reductions, repeated labels,
    or ellipses when exactly one input is an initializer. The initializer is
    reordered and reshaped to align its labels with the output axes so ONNX Mul
    broadcasting reproduces the Einsum result.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for node in graph.node:
            if node.op_type != "Einsum" or len(node.input) != 2:
                continue

            equation_attr = get_by_name(node.attribute, "equation")
            if equation_attr is None:
                continue
            equation = equation_attr.s.decode("UTF-8")
            try:
                input_subscripts, output_subscript = equation.split("->")
                input_subscripts = input_subscripts.split(",")
            except ValueError:
                continue

            if (
                len(input_subscripts) != 2
                or "..." in equation
                or any(len(set(subscript)) != len(subscript) for subscript in input_subscripts)
                or len(set(output_subscript)) != len(output_subscript)
                or set("".join(input_subscripts)) != set(output_subscript)
            ):
                continue

            initializer_index = next(
                (index for index, input_name in enumerate(node.input) if model.get_initializer(input_name) is not None),
                None,
            )
            if initializer_index is None:
                continue

            initializer_name = node.input[initializer_index]
            initializer = model.get_initializer(initializer_name)
            initializer_subscript = input_subscripts[initializer_index]
            if initializer.ndim != len(initializer_subscript):
                continue

            output_positions = [output_subscript.index(label) for label in initializer_subscript]
            permutation = np.argsort(output_positions)
            initializer = np.transpose(initializer, permutation)
            initializer = initializer.reshape(
                tuple(
                    initializer.shape[list(permutation).index(initializer_subscript.index(label))]
                    if label in initializer_subscript
                    else 1
                    for label in output_subscript
                )
            )
            model.set_initializer(initializer_name, initializer)
            node.op_type = "Mul"
            del node.attribute[:]
            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
        return (model, graph_modified)