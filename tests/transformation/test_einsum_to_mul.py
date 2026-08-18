import numpy as np
import pytest
from onnx import TensorProto, helper

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.einsum_to_mul import EinsumToMul
from qonnx.util.basic import qonnx_make_model


@pytest.mark.parametrize(
    "equation, parameter_shape, input_shape",
    [
        ("n,bn->bn", (4,), (2, 4)),
        ("fc,bcsf->bcsf", (2, 3), (2, 3, 5, 2)),
    ],
)
def test_einsum_to_mul_broadcast(equation, parameter_shape, input_shape):
    parameter = helper.make_tensor_value_info("parameter", TensorProto.FLOAT, parameter_shape)
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, input_shape)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shape)
    einsum = helper.make_node("Einsum", ["parameter", "data"], ["output"], equation=equation)
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([einsum], "einsum", [data], [output], value_info=[parameter]),
            opset_imports=[helper.make_opsetid("", 12)],
        )
    )
    model.set_initializer("parameter", np.arange(np.prod(parameter_shape), dtype=np.float32).reshape(parameter_shape))

    input_dict = {"data": np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)}
    expected = oxe.execute_onnx(model, input_dict)["output"]
    lowered_model = model.transform(EinsumToMul())

    assert [node.op_type for node in lowered_model.graph.node] == ["Mul"]
    produced = oxe.execute_onnx(lowered_model, input_dict)["output"]
    assert np.array_equal(expected, produced)


def test_einsum_to_mul_leaves_contractions_unchanged():
    weights = helper.make_tensor_value_info("weights", TensorProto.FLOAT, [3, 4])
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    einsum = helper.make_node("Einsum", ["data", "weights"], ["output"], equation="ab,bc->ac")
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph([einsum], "einsum", [data], [output], value_info=[weights]),
            opset_imports=[helper.make_opsetid("", 12)],
        )
    )
    model.set_initializer("weights", np.ones((3, 4), dtype=np.float32))

    lowered_model = model.transform(EinsumToMul())

    assert [node.op_type for node in lowered_model.graph.node] == ["Einsum"]