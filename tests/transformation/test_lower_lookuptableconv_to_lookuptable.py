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
from qonnx.custom_op.lnn.lookup_table_conv import lookup_table_conv
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_lookuptableconv_to_lookuptable import LowerLookupTableConvToLookupTable
from qonnx.util.basic import qonnx_make_model

DOMAIN = "qonnx.custom_op.lnn"

_DTYPE_MAP = {
    np.dtype(np.bool_): TensorProto.BOOL,
    np.dtype(np.uint8): TensorProto.UINT8,
}


def _ofm_dims(x_shape, kernel_shape, strides, pads):
    n_spatial = len(kernel_shape)
    return [
        (x_shape[1 + d] + pads[d] + pads[d + n_spatial] - kernel_shape[d]) // strides[d] + 1 for d in range(n_spatial)
    ]


def make_conv_model(x_shape, indices, table, tree_depth, kernel_shape, strides, pads):
    """table may be None (passthrough, tree_depth == 0 -> only X, indices as inputs)."""
    m, lut_rank = indices.shape[0], indices.shape[-1]
    ofm_dims = _ofm_dims(x_shape, kernel_shape, strides, pads)

    inp = helper.make_tensor_value_info("X", TensorProto.UINT8, list(x_shape))
    if tree_depth == 0:
        out_dtype, oshape = TensorProto.UINT8, [x_shape[0]] + ofm_dims + [m, lut_rank]
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
    )
    graph = helper.make_graph([node], "convlut_graph", [inp], [outp], initializer=initializers)
    model = qonnx_make_model(
        graph,
        producer_name="convlut-model",
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid(DOMAIN, 1),
            helper.make_opsetid("qonnx.custom_op.general", 1),
        ],
    )
    model = ModelWrapper(model)
    model.set_tensor_datatype("X", DataType["UINT8"])
    return model


def test_lower_tree_depth1():
    rng = np.random.default_rng(0)
    n, h, w, c, kh, kw, m, lut_rank = 1, 4, 4, 2, 2, 2, 3, 2
    patch = c * kh * kw
    indices = np.stack([rng.choice(patch, size=lut_rank, replace=False) for _ in range(m)]).astype(np.int64)
    indices = indices[:, None, :]  # (M, P=1, lut_rank)
    table = rng.integers(0, 2, size=(m, 1, 2**lut_rank)).astype(np.uint8)
    x = rng.integers(0, 2, size=(n, h, w, c)).astype(np.uint8)

    kernel_shape, strides, pads = [kh, kw], [2, 2], [0, 0, 0, 0]
    model = make_conv_model(x.shape, indices, table, 1, kernel_shape, strides, pads)
    expected = lookup_table_conv(x, indices, table, 1, kernel_shape, strides, pads)

    model = model.transform(LowerLookupTableConvToLookupTable())
    op_types = [n.op_type for n in model.graph.node]
    assert op_types == ["Im2Col", "LookupTable"]

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    assert model.get_tensor_shape("Y") == list(expected.shape)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_lower_tree_depth2_no_kernel_crosstalk():
    """Two kernels, depth-2 tree: verifies the canonical internal pairing never mixes
    nodes belonging to different kernels."""
    m, lut_rank = 2, 2
    indices = np.array(
        [
            [[0, 3], [1, 2]],  # kernel 0
            [[2, 1], [3, 0]],  # kernel 1
        ],
        dtype=np.int64,
    )  # (M=2, P=2, lut_rank=2)
    table = np.array(
        [
            [[0, 1, 1, 0], [0, 0, 0, 1], [0, 1, 1, 1]],  # kernel 0: XOR, AND, OR(leaf0,leaf1)
            [[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 0, 0]],  # kernel 1: XNOR, NAND, NOR(leaf0,leaf1)
        ],
        dtype=np.uint8,
    )
    x = np.array([[[[0], [1]], [[1], [0]]]], dtype=np.uint8)  # (1, 2, 2, 1)

    kernel_shape, strides, pads = [2, 2], [2, 2], [0, 0, 0, 0]
    model = make_conv_model(x.shape, indices, table, 2, kernel_shape, strides, pads)
    expected = lookup_table_conv(x, indices, table, 2, kernel_shape, strides, pads)

    model = model.transform(LowerLookupTableConvToLookupTable())
    op_types = [n.op_type for n in model.graph.node]
    assert op_types == ["Im2Col", "LookupTable", "LookupTable"]

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_lower_passthrough():
    indices = np.array([[[0, 3]]], dtype=np.int64)  # (M=1, P=1, lut_rank=2)
    x = np.array([[[[0], [1]], [[1], [0]]]], dtype=np.uint8)  # (1, 2, 2, 1)

    kernel_shape, strides, pads = [2, 2], [2, 2], [0, 0, 0, 0]
    model = make_conv_model(x.shape, indices, None, 0, kernel_shape, strides, pads)
    expected = lookup_table_conv(x, indices, None, 0, kernel_shape, strides, pads)

    model = model.transform(LowerLookupTableConvToLookupTable())
    op_types = [n.op_type for n in model.graph.node]
    assert op_types == ["Im2Col", "Gather"]

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_lower_preserves_graph_io():
    """The transformation must not change the graph's declared input/output tensor names."""
    indices = np.array([[[0, 3], [1, 2]]], dtype=np.int64)
    table = np.array([[[0, 1, 1, 0], [0, 0, 0, 1], [0, 1, 1, 1]]], dtype=np.uint8)
    x = np.array([[[[0], [1]], [[1], [0]]]], dtype=np.uint8)

    kernel_shape, strides, pads = [2, 2], [2, 2], [0, 0, 0, 0]
    model = make_conv_model(x.shape, indices, table, 2, kernel_shape, strides, pads)
    in_name, out_name = model.graph.input[0].name, model.graph.output[0].name

    model = model.transform(LowerLookupTableConvToLookupTable())
    assert model.graph.input[0].name == in_name
    assert model.graph.output[0].name == out_name
    assert not any(n.op_type == "LookupTableConv" for n in model.graph.node)


def test_lower_multi_channel_stride_padding():
    rng = np.random.default_rng(2)
    n, h, w, c, kh, kw, m, lut_rank = 1, 5, 5, 2, 3, 3, 3, 2
    patch = c * kh * kw
    indices = np.stack([rng.choice(patch, size=lut_rank, replace=False) for _ in range(m)]).astype(np.int64)
    indices = indices[:, None, :]
    table = rng.integers(0, 2, size=(m, 1, 2**lut_rank)).astype(np.uint8)
    x = rng.integers(0, 2, size=(n, h, w, c)).astype(np.uint8)

    kernel_shape, strides, pads = [kh, kw], [1, 1], [1, 1, 1, 1]
    model = make_conv_model(x.shape, indices, table, 1, kernel_shape, strides, pads)
    expected = lookup_table_conv(x, indices, table, 1, kernel_shape, strides, pads)

    model = model.transform(LowerLookupTableConvToLookupTable())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)
    assert produced.shape == (n, h, w, m)
