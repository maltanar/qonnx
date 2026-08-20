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
import pytest
from onnx import TensorProto, helper

import qonnx.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.intquant import int_quant
from qonnx.custom_op.lnn.lookup_table import lookup_table
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.change_batchsize import ChangeBatchSize
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

DOMAIN = "qonnx.custom_op.lnn"


def make_lut_model(x_shape, x_dtype, indices, table, input_bits=1, out_bits=0):
    m = indices.shape[0]
    inp = helper.make_tensor_value_info("X", x_dtype, x_shape)
    dtype_map = {
        np.dtype(np.bool_): TensorProto.BOOL,
        np.dtype(np.uint8): TensorProto.UINT8,
        np.dtype(np.int8): TensorProto.INT8,
        np.dtype(np.uint16): TensorProto.UINT16,
        np.dtype(np.int16): TensorProto.INT16,
        np.dtype(np.uint32): TensorProto.UINT32,
        np.dtype(np.int32): TensorProto.INT32,
    }
    outp = helper.make_tensor_value_info("Y", dtype_map[table.dtype], list(x_shape[:-1]) + [m])

    idx_init = helper.make_tensor(
        "indices", TensorProto.INT64, indices.shape, indices.flatten().astype(np.int64).tolist()
    )
    tab_init = helper.make_tensor("table", dtype_map[table.dtype], table.shape, table.flatten().tolist())

    node = helper.make_node(
        "LookupTable",
        ["X", "indices", "table"],
        ["Y"],
        domain=DOMAIN,
        input_bits=input_bits,
        out_bits=out_bits,
    )
    graph = helper.make_graph(
        [node], "lut_graph", [inp], [outp], initializer=[idx_init, tab_init]
    )
    model = qonnx_make_model(
        graph,
        producer_name="lut-model",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)],
    )
    model = ModelWrapper(model)
    return model


def test_execute_gatenet():
    """Two-input, one-bit gates: XOR, AND, OR over a 4-element binary input."""
    indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    table = np.array(
        [
            [0, 1, 1, 0],  # XOR
            [0, 0, 0, 1],  # AND
            [0, 1, 1, 1],  # OR
        ],
        dtype=np.uint8,
    )
    x = np.array([[0, 1, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)

    model = make_lut_model(list(x.shape), TensorProto.UINT8, indices, table)
    expected = lookup_table(x, indices, table)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_execute_sparse_layer():
    """Sparse layer with fan-in 6 over a 64-element uint8 input."""
    rng = np.random.default_rng(0)
    c_in, m, k = 64, 32, 6
    indices = np.stack([rng.choice(c_in, size=k, replace=False) for _ in range(m)]).astype(np.int64)
    table = rng.integers(0, 2, size=(m, 2**k)).astype(np.uint8)
    x = rng.integers(0, 2, size=(4, c_in)).astype(np.uint8)

    model = make_lut_model(list(x.shape), TensorProto.UINT8, indices, table)
    expected = lookup_table(x, indices, table)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_execute_multibit_input():
    """input_bits=2, fan-in 3 -> 6-bit addresses, table entries carry a 2-bit output."""
    rng = np.random.default_rng(1)
    c_in, m, k, ib = 8, 4, 3, 2
    indices = np.stack([rng.choice(c_in, size=k, replace=False) for _ in range(m)]).astype(np.int64)
    table = rng.integers(0, 4, size=(m, 2 ** (k * ib))).astype(np.uint8)
    x = rng.integers(0, 2**ib, size=(5, c_in)).astype(np.uint8)

    model = make_lut_model(list(x.shape), TensorProto.UINT8, indices, table, input_bits=ib, out_bits=2)
    expected = lookup_table(x, indices, table, input_bits=ib)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_execute_bool_io():
    indices = np.array([[0, 1]], dtype=np.int64)
    table = np.array([[False, True, True, False]], dtype=np.bool_)  # XOR
    x = np.array([[True, False], [True, True]], dtype=np.bool_)

    model = make_lut_model(list(x.shape), TensorProto.BOOL, indices, table)
    expected = lookup_table(x, indices, table)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_shape_inference():
    indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    table = np.zeros((3, 4), dtype=np.uint8)
    model = make_lut_model([2, 4], TensorProto.UINT8, indices, table)

    model = model.transform(InferShapes())
    assert model.get_tensor_shape("Y") == [2, 3]


def test_datatype_inference_unsigned():
    indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    table = np.zeros((3, 4), dtype=np.uint8)
    model = make_lut_model([None, 4], TensorProto.UINT8, indices, table)

    model = model.transform(InferDataTypes())
    # no out_bits set -> full 8-bit container width
    assert model.get_tensor_datatype("Y") == DataType["UINT8"]


def test_datatype_inference_out_bits():
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    table = np.zeros((1, 8), dtype=np.uint8)
    model = make_lut_model([None, 3], TensorProto.UINT8, indices, table, out_bits=2)

    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("Y") == DataType["UINT2"]


def test_datatype_inference_bool_is_binary():
    indices = np.array([[0, 1]], dtype=np.int64)
    table = np.zeros((1, 4), dtype=np.bool_)
    model = make_lut_model([None, 2], TensorProto.BOOL, indices, table)

    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("Y") == DataType["BINARY"]


def test_datatype_inference_signed():
    indices = np.array([[0, 1]], dtype=np.int64)
    table = np.zeros((1, 4), dtype=np.int8)
    model = make_lut_model([None, 2], TensorProto.INT8, indices, table)

    model = model.transform(InferDataTypes())
    assert model.get_tensor_datatype("Y") == DataType["INT8"]


def test_verify_node():
    indices = np.array([[0, 1]], dtype=np.int64)
    table = np.zeros((1, 4), dtype=np.uint8)
    model = make_lut_model([None, 2], TensorProto.UINT8, indices, table)
    node = model.graph.node[0]
    inst = getCustomOp(node)
    messages = inst.verify_node()
    assert any("All necessary attributes exist" in m for m in messages)
    assert any("number of inputs is correct" in m for m in messages)


def test_chained_lookup_tables_survive_change_batchsize():
    """Regression test: two chained LookupTable nodes used to break InferShapes after
    ChangeBatchSize wipes all intermediate ValueInfo, since the second node's
    make_shape_compatible_op couldn't resolve the first node's (also-just-hidden) output shape.
    """
    c_in, m1, m2, k = 8, 5, 3, 2
    rng = np.random.default_rng(5)
    idx1 = np.stack([rng.choice(c_in, size=k, replace=False) for _ in range(m1)]).astype(np.int64)
    tab1 = rng.integers(0, 2, size=(m1, 2**k)).astype(np.uint8)
    idx2 = np.stack([rng.choice(m1, size=k, replace=False) for _ in range(m2)]).astype(np.int64)
    tab2 = rng.integers(0, 2, size=(m2, 2**k)).astype(np.bool_)  # second layer outputs bool

    inp = helper.make_tensor_value_info("X", TensorProto.UINT8, [1, c_in])
    outp = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, m2])
    idx1_init = helper.make_tensor("idx1", TensorProto.INT64, idx1.shape, idx1.flatten().tolist())
    tab1_init = helper.make_tensor("tab1", TensorProto.UINT8, tab1.shape, tab1.flatten().tolist())
    idx2_init = helper.make_tensor("idx2", TensorProto.INT64, idx2.shape, idx2.flatten().tolist())
    tab2_init = helper.make_tensor("tab2", TensorProto.BOOL, tab2.shape, tab2.flatten().tolist())

    nodes = [
        helper.make_node("LookupTable", ["X", "idx1", "tab1"], ["hidden"], domain=DOMAIN),
        helper.make_node("LookupTable", ["hidden", "idx2", "tab2"], ["Y"], domain=DOMAIN),
    ]
    graph = helper.make_graph(
        nodes, "chained_lut_graph", [inp], [outp], initializer=[idx1_init, tab1_init, idx2_init, tab2_init]
    )
    model = qonnx_make_model(
        graph,
        producer_name="chained-lut-model",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)],
    )
    model = ModelWrapper(model)

    batch_size = 100
    model = model.transform(ChangeBatchSize(batch_size))
    model = model.transform(InferShapes())
    assert model.get_tensor_shape("hidden") == [batch_size, m1]
    assert model.get_tensor_shape("Y") == [batch_size, m2]

    x = rng.integers(0, 2, size=(batch_size, c_in)).astype(np.uint8)
    hidden = lookup_table(x, idx1, tab1)
    expected = lookup_table(hidden, idx2, tab2)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def _im2col_nhwc(x, k):
    """Reference im2col matching qonnx's Im2Col (NHWC, stride 1, no padding)."""
    n, h, w, c = x.shape
    oh, ow = h - k + 1, w - k + 1
    out = np.zeros((n, oh, ow, k * k * c), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            out[:, i, j, :] = x[:, i : i + k, j : j + k, :].reshape(n, -1)
    return out


def test_conv_with_im2col():
    """Convolutional LNN: Im2Col (NHWC) feeding LookupTable over the patch axis, both real qonnx CustomOps."""
    rng = np.random.default_rng(3)
    n, h, w, c, k, m = 1, 6, 6, 2, 3, 4
    patch = k * k * c
    indices = np.stack([rng.choice(patch, size=4, replace=False) for _ in range(m)]).astype(np.int64)
    table = rng.integers(0, 2, size=(m, 2**4)).astype(np.uint8)

    x = rng.integers(0, 2, size=(n, h, w, c)).astype(np.uint8)
    oh, ow = h - k + 1, w - k + 1

    inp = helper.make_tensor_value_info("X", TensorProto.UINT8, [n, h, w, c])
    outp = helper.make_tensor_value_info("Y", TensorProto.UINT8, [n, oh, ow, m])
    idx_init = helper.make_tensor("indices", TensorProto.INT64, indices.shape, indices.flatten().tolist())
    tab_init = helper.make_tensor("table", TensorProto.UINT8, table.shape, table.flatten().tolist())

    im2col = helper.make_node(
        "Im2Col",
        ["X"],
        ["patches"],
        domain="qonnx.custom_op.general",
        stride=[1, 1],
        kernel_size=[k, k],
        input_shape=str((n, h, w, c)),
        pad_amount=[0, 0, 0, 0],
    )
    lut = helper.make_node("LookupTable", ["patches", "indices", "table"], ["Y"], domain=DOMAIN)
    # declared explicitly since qonnx's InferShapes cannot chain two custom ops without ONNX
    # shape inference running in between; here we sidestep it and specify it directly
    patches_vi = helper.make_tensor_value_info("patches", TensorProto.UINT8, [n, oh, ow, patch])
    graph = helper.make_graph(
        [im2col, lut], "conv_lut_graph", [inp], [outp], initializer=[idx_init, tab_init], value_info=[patches_vi]
    )
    model = qonnx_make_model(
        graph,
        producer_name="conv-lut-model",
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("qonnx.custom_op.general", 1),
            helper.make_opsetid(DOMAIN, 1),
        ],
    )
    model = ModelWrapper(model)
    model.set_tensor_datatype("X", DataType["UINT8"])

    patches = _im2col_nhwc(x, k)
    expected = lookup_table(patches, indices, table)

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.array_equal(produced, expected)


def test_hybrid_quant_lut():
    """Quant -> Cast into LookupTable's integer domain -> Cast+Mul for a scaled output."""
    rng = np.random.default_rng(4)
    c_in, m, k, b = 8, 4, 3, 2
    scale = np.float32(0.25)
    indices = np.stack([rng.choice(c_in, size=k, replace=False) for _ in range(m)]).astype(np.int64)
    table = rng.integers(0, 4, size=(m, 2 ** (k * b))).astype(np.uint8)  # 6:2 LUTs, 2-bit output packed in uint8
    out_scale = np.float32(0.5)

    n = 4
    inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, c_in])
    outp = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, m])
    idx_init = helper.make_tensor("indices", TensorProto.INT64, indices.shape, indices.flatten().tolist())
    tab_init = helper.make_tensor("table", TensorProto.UINT8, table.shape, table.flatten().tolist())
    scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [], [scale])
    zeropt_init = helper.make_tensor("zeropt", TensorProto.FLOAT, [], [0.0])
    bitwidth_init = helper.make_tensor("bitwidth", TensorProto.FLOAT, [], [float(b)])
    out_scale_init = helper.make_tensor("out_scale", TensorProto.FLOAT, [], [out_scale])

    nodes = [
        helper.make_node(
            "Quant",
            ["X", "scale", "zeropt", "bitwidth"],
            ["Xq"],
            domain="qonnx.custom_op.general",
            signed=0,
            narrow=0,
            rounding_mode="ROUND",
        ),
        helper.make_node("Div", ["Xq", "scale"], ["Xint"]),
        helper.make_node("Cast", ["Xint"], ["Xu"], to=TensorProto.UINT8),
        helper.make_node(
            "LookupTable", ["Xu", "indices", "table"], ["Ylut"], domain=DOMAIN, input_bits=b, out_bits=2
        ),
        helper.make_node("Cast", ["Ylut"], ["Ylutf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["Ylutf", "out_scale"], ["Y"]),
    ]
    # declared explicitly since qonnx's InferShapes cannot chain two custom ops (Quant, LookupTable)
    # without ONNX shape inference running in between; here we sidestep it and specify it directly
    value_info = [
        helper.make_tensor_value_info("Xq", TensorProto.FLOAT, [n, c_in]),
        helper.make_tensor_value_info("Xint", TensorProto.FLOAT, [n, c_in]),
        helper.make_tensor_value_info("Xu", TensorProto.UINT8, [n, c_in]),
        helper.make_tensor_value_info("Ylut", TensorProto.UINT8, [n, m]),
        helper.make_tensor_value_info("Ylutf", TensorProto.FLOAT, [n, m]),
    ]
    graph = helper.make_graph(
        nodes,
        "hybrid_quant_graph",
        [inp],
        [outp],
        initializer=[idx_init, tab_init, scale_init, zeropt_init, bitwidth_init, out_scale_init],
        value_info=value_info,
    )
    model = qonnx_make_model(
        graph,
        producer_name="hybrid-quant-model",
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("qonnx.custom_op.general", 1),
            helper.make_opsetid(DOMAIN, 1),
        ],
    )
    model = ModelWrapper(model)

    x = rng.uniform(0, 0.75, size=(n, c_in)).astype(np.float32)
    xq = int_quant(x, scale, np.float32(0.0), np.float32(b), signed=0, narrow=0, rounding_mode="ROUND")
    xint = (xq / scale).astype(np.uint8)
    expected = lookup_table(xint, indices, table, input_bits=b).astype(np.float32) * out_scale

    produced = oxe.execute_onnx(model, {"X": x})["Y"]
    assert np.allclose(produced, expected)


@pytest.mark.parametrize("k", [2, 3, 6])
def test_reference_matches_brute_force(k):
    """Cross-check the vectorized numpy reference against a brute-force per-neuron loop."""
    rng = np.random.default_rng(2)
    m, c_in = 5, 10
    indices = np.stack([rng.choice(c_in, size=k, replace=False) for _ in range(m)]).astype(np.int64)
    table = rng.integers(0, 2, size=(m, 2**k)).astype(np.uint8)
    x = rng.integers(0, 2, size=(3, c_in)).astype(np.uint8)

    got = lookup_table(x, indices, table)

    expected = np.zeros((3, m), dtype=np.uint8)
    for n in range(3):
        for mi in range(m):
            addr = 0
            for kk in range(k):
                addr |= int(x[n, indices[mi, kk]]) << kk
            expected[n, mi] = table[mi, addr]
    assert np.array_equal(got, expected)
