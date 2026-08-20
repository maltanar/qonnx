# Copyright (c) 2026 EmLogic AS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of EmLogic nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from onnx import TensorProto, helper

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.lnn.lookup_table import lookup_table
from qonnx.custom_op.lnn.lookup_table_conv import lookup_table_conv
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

DOMAIN = "qonnx.custom_op.lnn"

_DTYPE_MAP = {
    np.dtype(np.bool_): TensorProto.BOOL,
    np.dtype(np.uint8): TensorProto.UINT8,
    np.dtype(np.int8): TensorProto.INT8,
    np.dtype(np.uint16): TensorProto.UINT16,
    np.dtype(np.int16): TensorProto.INT16,
    np.dtype(np.uint32): TensorProto.UINT32,
    np.dtype(np.int32): TensorProto.INT32,
}


def _ofm_dims(x_shape, kernel_shape, strides, pads):
    n_spatial = len(kernel_shape)
    return [
        (x_shape[1 + d] + pads[d] + pads[d + n_spatial] - kernel_shape[d]) // strides[d] + 1 for d in range(n_spatial)
    ]


def make_conv_model(
    x_shape, x_dtype, indices, table, tree_depth, kernel_shape, strides, pads, channel_group_size=0
):
    """table may be None (passthrough, tree_depth == 0 -> only X, indices as inputs)."""
    m, lut_rank = indices.shape[0], indices.shape[-1]
    ofm_dims = _ofm_dims(x_shape, kernel_shape, strides, pads)

    inp = helper.make_tensor_value_info("X", x_dtype, list(x_shape))
    if tree_depth == 0:
        out_dtype, oshape = x_dtype, [x_shape[0]] + ofm_dims + [m, lut_rank]
    else:
        out_dtype, oshape = _DTYPE_MAP[table.dtype], [x_shape[0]] + ofm_dims + [m]
    outp = helper.make_tensor_value_info("Y", out_dtype, oshape)

    idx_init = helper.make_tensor(
        "indices", TensorProto.INT64, indices.shape, indices.flatten().astype(np.int64).tolist()
    )
    node_inputs = ["X", "indices"]
    initializers = [idx_init]
    if table is not None:
        tab_init = helper.make_tensor("table", _DTYPE_MAP[table.dtype], table.shape, table.flatten().tolist())
        node_inputs.append("table")
        initializers.append(tab_init)

    node = helper.make_node(
        "LookupTableConv",
        node_inputs,
        ["Y"],
        domain=DOMAIN,
        tree_depth=tree_depth,
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        channel_group_size=channel_group_size,
    )
    graph = helper.make_graph([node], "convlut_graph", [inp], [outp], initializer=initializers)
    model = qonnx_make_model(
        graph,
        producer_name="convlut-model",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)],
    )
    return ModelWrapper(model)


def test_execute_single_level():
    """tree_depth=1: a 2x2 receptive field, stride 2, 1 channel, two 2-input gates (XOR, AND)."""
    indices = np.array([[[0, 3]], [[1, 2]]], dtype=np.int64)  # (M=2, P=1, lut_rank=2)
    table = np.array([[[0, 1, 1, 0]], [[0, 0, 0, 1]]], dtype=np.uint8)  # XOR, AND; (M=2, N_nodes=1, 4)
    x = np.array([[[[0], [1]], [[1], [0]]]], dtype=np.uint8)  # (1, 2, 2, 1)

    model = make_conv_model(x.shape, TensorProto.UINT8, indices, table, 1, [2, 2], [2, 2], [0, 0, 0, 0])
    expected = lookup_table_conv(x, indices, table, 1, [2, 2], [2, 2], [0, 0, 0, 0])

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_execute_tree_depth2():
    """Depth-2 tree of 2-input gates: two leaf gates feed a root gate, over a 2x2 receptive field."""
    indices = np.array([[[0, 3], [1, 2]]], dtype=np.int64)  # (M=1, P=2, lut_rank=2)
    table = np.array(
        [
            [
                [0, 1, 1, 0],  # leaf 0: XOR
                [0, 0, 0, 1],  # leaf 1: AND
                [0, 1, 1, 1],  # root: OR(leaf0, leaf1)
            ]
        ],
        dtype=np.uint8,
    )
    x = np.array([[[[0], [1]], [[1], [0]]]], dtype=np.uint8)  # (1, 2, 2, 1)

    model = make_conv_model(x.shape, TensorProto.UINT8, indices, table, 2, [2, 2], [2, 2], [0, 0, 0, 0])
    expected = lookup_table_conv(x, indices, table, 2, [2, 2], [2, 2], [0, 0, 0, 0])

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)
    # XOR(0,1)=1, AND(1,0)=0, OR(1,0)=1
    assert np.array_equal(produced, np.array([[[[1]]]], dtype=np.uint8))


def test_execute_multi_channel_stride_padding():
    """2 input channels, 3x3 receptive field, stride 1, symmetric padding of 1."""
    rng = np.random.default_rng(0)
    n, h, w, c, kh, kw, m, lut_rank = 2, 5, 5, 2, 3, 3, 4, 4
    patch = c * kh * kw
    indices = np.stack([rng.choice(patch, size=lut_rank, replace=False) for _ in range(m)]).astype(np.int64)
    indices = indices[:, None, :]  # (M, P=1, lut_rank)
    table = rng.integers(0, 2, size=(m, 1, 2**lut_rank)).astype(np.uint8)
    x = rng.integers(0, 2, size=(n, h, w, c)).astype(np.uint8)

    kernel_shape, strides, pads = [kh, kw], [1, 1], [1, 1, 1, 1]
    model = make_conv_model(x.shape, TensorProto.UINT8, indices, table, 1, kernel_shape, strides, pads)
    expected = lookup_table_conv(x, indices, table, 1, kernel_shape, strides, pads)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)
    assert produced.shape == (n, h, w, m)  # stride 1 + padding 1 with a 3x3 kernel preserves spatial dims


def test_execute_passthrough():
    """tree_depth=0: table omitted, raw fan-in gathered from the receptive field."""
    indices = np.array([[[0, 3]]], dtype=np.int64)  # (M=1, P=1, lut_rank=2)
    x = np.array([[[[0], [1]], [[1], [0]]]], dtype=np.uint8)  # (1, 2, 2, 1)

    model = make_conv_model(x.shape, TensorProto.UINT8, indices, None, 0, [2, 2], [2, 2], [0, 0, 0, 0])
    expected = lookup_table_conv(x, indices, None, 0, [2, 2], [2, 2], [0, 0, 0, 0])

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)
    # flattened patch is [0, 1, 1, 0]; indices [0, 3] -> gathered values [0, 0]
    assert np.array_equal(produced, np.array([[[[[0, 0]]]]], dtype=np.uint8))


def test_shape_inference_tree():
    indices = np.array([[[0, 3], [1, 2]]], dtype=np.int64)
    table = np.zeros((1, 3, 4), dtype=np.uint8)
    model = make_conv_model([1, 4, 4, 1], TensorProto.UINT8, indices, table, 2, [2, 2], [2, 2], [0, 0, 0, 0])

    model = model.transform(InferShapes())
    assert model.get_tensor_shape("Y") == [1, 2, 2, 1]


def test_shape_inference_passthrough():
    indices = np.array([[[0, 3]]], dtype=np.int64)
    model = make_conv_model([1, 2, 2, 1], TensorProto.UINT8, indices, None, 0, [2, 2], [2, 2], [0, 0, 0, 0])

    model = model.transform(InferShapes())
    assert model.get_tensor_shape("Y") == [1, 1, 1, 1, 2]


def test_datatype_inference_table_uint8():
    indices = np.array([[[0, 3], [1, 2]]], dtype=np.int64)
    table = np.zeros((1, 3, 4), dtype=np.uint8)
    model = make_conv_model([None, 4, 4, 1], TensorProto.UINT8, indices, table, 2, [2, 2], [2, 2], [0, 0, 0, 0])

    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("Y") == DataType["UINT8"]


def test_datatype_inference_table_bool_is_binary():
    indices = np.array([[[0, 3], [1, 2]]], dtype=np.int64)
    table = np.zeros((1, 3, 4), dtype=np.bool_)
    model = make_conv_model([None, 4, 4, 1], TensorProto.BOOL, indices, table, 2, [2, 2], [2, 2], [0, 0, 0, 0])

    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("Y") == DataType["BINARY"]


def test_datatype_inference_passthrough_forwards_input():
    indices = np.array([[[0, 3]]], dtype=np.int64)
    model = make_conv_model([None, 2, 2, 1], TensorProto.UINT8, indices, None, 0, [2, 2], [2, 2], [0, 0, 0, 0])
    model.set_tensor_datatype("X", DataType["BINARY"])

    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("Y") == DataType["BINARY"]


def test_verify_node_tree():
    indices = np.array([[[0, 3], [1, 2]]], dtype=np.int64)
    table = np.zeros((1, 3, 4), dtype=np.uint8)
    model = make_conv_model([None, 4, 4, 1], TensorProto.UINT8, indices, table, 2, [2, 2], [2, 2], [0, 0, 0, 0])
    inst = getCustomOp(model.graph.node[0])
    messages = inst.verify_node()
    assert any("All necessary attributes exist" in msg for msg in messages)
    assert any("number of inputs is correct" in msg for msg in messages)


def test_verify_node_passthrough():
    indices = np.array([[[0, 3]]], dtype=np.int64)
    model = make_conv_model([None, 2, 2, 1], TensorProto.UINT8, indices, None, 0, [2, 2], [2, 2], [0, 0, 0, 0])
    inst = getCustomOp(model.graph.node[0])
    messages = inst.verify_node()
    assert any("All necessary attributes exist" in msg for msg in messages)
    assert any("number of inputs is correct" in msg for msg in messages)


def test_conv_then_dense_lookup_table_chain():
    """LookupTableConv (tree_depth=1) feeding a plain LookupTable, mimicking a conv layer
    followed by a fully-connected layer over its M output channels."""
    rng = np.random.default_rng(1)
    n, h, w, m1, m2, lut_rank = 1, 4, 4, 3, 2, 2
    conv_indices = np.stack([rng.choice(4, size=lut_rank, replace=False) for _ in range(m1)]).astype(np.int64)
    conv_indices = conv_indices[:, None, :]
    conv_table = rng.integers(0, 2, size=(m1, 1, 2**lut_rank)).astype(np.uint8)
    dense_indices = np.stack([rng.choice(m1, size=lut_rank, replace=False) for _ in range(m2)]).astype(np.int64)
    dense_table = rng.integers(0, 2, size=(m2, 2**lut_rank)).astype(np.uint8)

    x = rng.integers(0, 2, size=(n, h, w, 1)).astype(np.uint8)
    kernel_shape, strides, pads = [2, 2], [2, 2], [0, 0, 0, 0]
    ofm_dims = _ofm_dims(x.shape, kernel_shape, strides, pads)

    inp = helper.make_tensor_value_info("X", TensorProto.UINT8, list(x.shape))
    outp = helper.make_tensor_value_info("Y", TensorProto.UINT8, [n] + ofm_dims + [m2])
    conv_idx_init = helper.make_tensor(
        "conv_idx", TensorProto.INT64, conv_indices.shape, conv_indices.flatten().tolist()
    )
    conv_tab_init = helper.make_tensor(
        "conv_tab", TensorProto.UINT8, conv_table.shape, conv_table.flatten().tolist()
    )
    dense_idx_init = helper.make_tensor(
        "dense_idx", TensorProto.INT64, dense_indices.shape, dense_indices.flatten().tolist()
    )
    dense_tab_init = helper.make_tensor(
        "dense_tab", TensorProto.UINT8, dense_table.shape, dense_table.flatten().tolist()
    )

    nodes = [
        helper.make_node(
            "LookupTableConv",
            ["X", "conv_idx", "conv_tab"],
            ["hidden"],
            domain=DOMAIN,
            tree_depth=1,
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
        ),
        helper.make_node("LookupTable", ["hidden", "dense_idx", "dense_tab"], ["Y"], domain=DOMAIN),
    ]
    # declared explicitly since qonnx's InferShapes cannot chain two custom ops without ONNX
    # shape inference running in between; here we sidestep it and specify it directly
    hidden_vi = helper.make_tensor_value_info("hidden", TensorProto.UINT8, [n] + ofm_dims + [m1])
    graph = helper.make_graph(
        nodes,
        "conv_then_dense_graph",
        [inp],
        [outp],
        initializer=[conv_idx_init, conv_tab_init, dense_idx_init, dense_tab_init],
        value_info=[hidden_vi],
    )
    model = qonnx_make_model(
        graph,
        producer_name="conv-then-dense-model",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)],
    )
    model = ModelWrapper(model)

    hidden = lookup_table_conv(x, conv_indices, conv_table, 1, kernel_shape, strides, pads)
    expected = lookup_table(hidden, dense_indices, dense_table)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)
