# Copyright (c) 2026 EmLogic
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
import onnx.helper as oh
from onnx import TensorProto

from qonnx.analysis.symbolic_dependencies import symbolic_dependencies
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model


def test_symbolic_dependencies_matmul_sparse_weight():
    x = oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("MatMul", ["x", "W"], ["y"]),
        ],
        name="symdep_matmul_sparse",
        inputs=[x],
        outputs=[y],
        value_info=[],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("W", np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32))

    ret = model.analysis(symbolic_dependencies)
    tdeps = ret["symbolic_dependencies"]["tensor_dependencies"]
    ydeps = tdeps["y"]

    assert ydeps[0] == {("x", 0)}
    assert ydeps[1] == {("x", 1)}


def test_symbolic_dependencies_quantizer_boundary():
    x = oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [2])
    relu_out = oh.make_tensor_value_info("relu_out", TensorProto.FLOAT, [2])
    q_out = oh.make_tensor_value_info("q_out", TensorProto.FLOAT, [2])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Relu", ["x"], ["relu_out"]),
            oh.make_node(
                "Quant",
                ["relu_out", "scale"],
                ["q_out"],
                domain="qonnx.custom_op.general",
            ),
            oh.make_node("Add", ["q_out", "c"], ["out"]),
        ],
        name="symdep_quant_boundary",
        inputs=[x],
        outputs=[out],
        value_info=[relu_out, q_out],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("scale", np.asarray(1.0, dtype=np.float32))
    model.set_initializer("c", np.asarray([1.0, 1.0], dtype=np.float32))
    model.save("symdep_quant_boundary.onnx")

    ret_stop = model.analysis(symbolic_dependencies)
    tdeps_stop = ret_stop["symbolic_dependencies"]["tensor_dependencies"]
    node_local_stop = ret_stop["symbolic_dependencies"]["node_local_dependencies"]
    assert tdeps_stop["q_out"][0] == {("q_out", 0)}
    assert tdeps_stop["q_out"][1] == {("q_out", 1)}
    assert tdeps_stop["out"][0] == {("q_out", 0)}
    assert tdeps_stop["out"][1] == {("q_out", 1)}

    assert node_local_stop["0:Relu"]["relu_out"] == [{("x", 0)}, {("x", 1)}]
    assert node_local_stop["1:Quant"]["q_out"] == [{("relu_out", 0)}, {("relu_out", 1)}]
    assert node_local_stop["2:Add"]["out"] == [{("q_out", 0)}, {("q_out", 1)}]


def test_symbolic_dependencies_expand_tile_reducesum():
    x = oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [2])
    expanded = oh.make_tensor_value_info("expanded", TensorProto.FLOAT, [2, 2])
    tiled = oh.make_tensor_value_info("tiled", TensorProto.FLOAT, [2, 4])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Expand", ["x", "exp_shape"], ["expanded"]),
            oh.make_node("Tile", ["expanded", "repeats"], ["tiled"]),
            oh.make_node("ReduceSum", ["tiled"], ["out"], axes=[1], keepdims=0),
        ],
        name="symdep_expand_tile_reducesum",
        inputs=[x],
        outputs=[out],
        value_info=[expanded, tiled],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("exp_shape", np.asarray([2, 2], dtype=np.int64))
    model.set_initializer("repeats", np.asarray([1, 2], dtype=np.int64))

    ret = model.analysis(symbolic_dependencies)
    tdeps = ret["symbolic_dependencies"]["tensor_dependencies"]
    out_deps = tdeps["out"]

    assert out_deps[0] == {("x", 0), ("x", 1)}
    assert out_deps[1] == {("x", 0), ("x", 1)}


def test_symbolic_dependencies_batchnorm_relu_passthrough():
    x = oh.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [2])
    id0_out = oh.make_tensor_value_info("id0_out", TensorProto.FLOAT, [2])
    bn_out = oh.make_tensor_value_info("bn_out", TensorProto.FLOAT, [2])
    relu_out = oh.make_tensor_value_info("relu_out", TensorProto.FLOAT, [2])
    id1_out = oh.make_tensor_value_info("id1_out", TensorProto.FLOAT, [2])
    id2_out = oh.make_tensor_value_info("id2_out", TensorProto.FLOAT, [2])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Identity", ["x"], ["id0_out"]),
            oh.make_node(
                "BatchNormalization",
                ["id0_out", "scale", "bias", "mean", "var"],
                ["bn_out"],
                epsilon=1e-5,
            ),
            oh.make_node("Relu", ["bn_out"], ["relu_out"]),
            oh.make_node("Identity", ["relu_out"], ["id1_out"]),
            oh.make_node("Identity", ["id1_out"], ["id2_out"]),
            oh.make_node("Identity", ["id2_out"], ["out"]),
        ],
        name="symdep_bn_relu",
        inputs=[x],
        outputs=[out],
        value_info=[id0_out, bn_out, relu_out, id1_out, id2_out],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.save("symdep_bn_relu.onnx")
    model.set_initializer("scale", np.asarray([1.0, 1.0], dtype=np.float32))
    model.set_initializer("bias", np.asarray([0.0, 0.0], dtype=np.float32))
    model.set_initializer("mean", np.asarray([0.0, 0.0], dtype=np.float32))
    model.set_initializer("var", np.asarray([1.0, 1.0], dtype=np.float32))

    ret = model.analysis(symbolic_dependencies)
    symdeps = ret["symbolic_dependencies"]
    tdeps = symdeps["tensor_dependencies"]
    node_local = symdeps["node_local_dependencies"]

    expected = [{("x", 0)}, {("x", 1)}]
    for tensor_name in ["id0_out", "bn_out", "relu_out", "id1_out", "id2_out", "out"]:
        assert tdeps[tensor_name] == expected

    assert node_local["0:Identity"]["id0_out"] == [{("x", 0)}, {("x", 1)}]
    assert node_local["1:BatchNormalization"]["bn_out"] == [{("id0_out", 0)}, {("id0_out", 1)}]
    assert node_local["2:Relu"]["relu_out"] == [{("bn_out", 0)}, {("bn_out", 1)}]
    assert node_local["3:Identity"]["id1_out"] == [{("relu_out", 0)}, {("relu_out", 1)}]
    assert node_local["4:Identity"]["id2_out"] == [{("id1_out", 0)}, {("id1_out", 1)}]
    assert node_local["5:Identity"]["out"] == [{("id2_out", 0)}, {("id2_out", 1)}]


def test_symbolic_dependencies_slice_gather_concat():
    x = oh.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [4])
    id_out = oh.make_tensor_value_info("id_out", TensorProto.FLOAT, [4])
    sliced = oh.make_tensor_value_info("sliced", TensorProto.FLOAT, [2])
    gathered = oh.make_tensor_value_info("gathered", TensorProto.FLOAT, [2])

    graph = oh.make_graph(
        nodes=[
            oh.make_node("Identity", ["x"], ["id_out"]),
            oh.make_node("Slice", ["id_out", "starts", "ends", "axes", "steps"], ["sliced"]),
            oh.make_node("Gather", ["sliced", "indices"], ["gathered"], axis=0),
            oh.make_node("Concat", ["gathered", "gathered"], ["out"], axis=0),
        ],
        name="symdep_slice_gather_concat",
        inputs=[x],
        outputs=[out],
        value_info=[id_out, sliced, gathered],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("starts", np.asarray([1], dtype=np.int64))
    model.set_initializer("ends", np.asarray([3], dtype=np.int64))
    model.set_initializer("axes", np.asarray([0], dtype=np.int64))
    model.set_initializer("steps", np.asarray([1], dtype=np.int64))
    model.set_initializer("indices", np.asarray([1, 0], dtype=np.int64))
    model.save("symdep_slice_gather_concat.onnx")

    ret = model.analysis(symbolic_dependencies)
    tdeps = ret["symbolic_dependencies"]["tensor_dependencies"]
    node_local = ret["symbolic_dependencies"]["node_local_dependencies"]
    out_deps = tdeps["out"]

    assert out_deps[0] == {("x", 2)}
    assert out_deps[1] == {("x", 1)}
    assert out_deps[2] == {("x", 2)}
    assert out_deps[3] == {("x", 1)}

    assert node_local["0:Identity"]["id_out"] == [{("x", 0)}, {("x", 1)}, {("x", 2)}, {("x", 3)}]
    assert node_local["1:Slice"]["sliced"] == [{("id_out", 1)}, {("id_out", 2)}]
    assert node_local["2:Gather"]["gathered"] == [{("sliced", 1)}, {("sliced", 0)}]
    assert node_local["3:Concat"]["out"] == [{("gathered", 0)}, {("gathered", 1)}, {("gathered", 0)}, {("gathered", 1)}]


def test_symbolic_dependencies_small_lookup_like_chain():
    x = oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6])
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, [1])

    q0 = oh.make_tensor_value_info("q0", TensorProto.FLOAT, [1, 6])
    gathered = oh.make_tensor_value_info("gathered", TensorProto.FLOAT, [1, 2])
    expanded = oh.make_tensor_value_info("expanded", TensorProto.FLOAT, [1, 2, 2])
    tiled = oh.make_tensor_value_info("tiled", TensorProto.FLOAT, [1, 2, 4])
    reshaped = oh.make_tensor_value_info("reshaped", TensorProto.FLOAT, [1, 8])
    multiplied = oh.make_tensor_value_info("multiplied", TensorProto.FLOAT, [1, 8])
    reduced = oh.make_tensor_value_info("reduced", TensorProto.FLOAT, [1])
    shifted = oh.make_tensor_value_info("shifted", TensorProto.FLOAT, [1])
    relu_out = oh.make_tensor_value_info("relu_out", TensorProto.FLOAT, [1])
    bn_out = oh.make_tensor_value_info("bn_out", TensorProto.FLOAT, [1])
    res_mul = oh.make_tensor_value_info("res_mul", TensorProto.FLOAT, [1, 2])
    res_red = oh.make_tensor_value_info("res_red", TensorProto.FLOAT, [1])
    res_shifted = oh.make_tensor_value_info("res_shifted", TensorProto.FLOAT, [1])
    merged = oh.make_tensor_value_info("merged", TensorProto.FLOAT, [1])

    graph = oh.make_graph(
        nodes=[
            oh.make_node(
                "BipolarQuant",
                ["x", "scale0"],
                ["q0"],
                domain="qonnx.custom_op.general",
            ),
            oh.make_node("Gather", ["q0", "indices"], ["gathered"], axis=1),
            oh.make_node("Expand", ["gathered", "exp_shape"], ["expanded"]),
            oh.make_node("Tile", ["expanded", "repeats"], ["tiled"]),
            oh.make_node("Reshape", ["tiled", "reshape_shape"], ["reshaped"]),
            oh.make_node("Mul", ["reshaped", "mul_mask"], ["multiplied"]),
            oh.make_node("ReduceSum", ["multiplied"], ["reduced"], axes=[1], keepdims=0),
            oh.make_node("Add", ["reduced", "bias"], ["shifted"]),
            oh.make_node("Relu", ["shifted"], ["relu_out"]),
            oh.make_node("Mul", ["gathered", "res_mask"], ["res_mul"]),
            oh.make_node("ReduceSum", ["res_mul"], ["res_red"], axes=[1], keepdims=0),
            oh.make_node("Add", ["res_red", "res_bias"], ["res_shifted"]),
            oh.make_node("Add", ["relu_out", "res_shifted"], ["merged"]),
            oh.make_node(
                "BatchNormalization",
                ["merged", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                ["bn_out"],
                epsilon=1e-5,
            ),
            oh.make_node(
                "BipolarQuant",
                ["bn_out", "scale1"],
                ["out"],
                domain="qonnx.custom_op.general",
            ),
        ],
        name="symdep_small_lookup_like_chain",
        inputs=[x],
        outputs=[out],
        value_info=[q0, gathered, expanded, tiled, reshaped, multiplied, reduced, shifted, relu_out, res_mul, res_red, res_shifted, merged, bn_out],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("scale0", np.asarray(1.0, dtype=np.float32))
    model.set_initializer("indices", np.asarray([3, 1], dtype=np.int64))
    model.set_initializer("exp_shape", np.asarray([1, 2, 2], dtype=np.int64))
    model.set_initializer("repeats", np.asarray([1, 1, 2], dtype=np.int64))
    model.set_initializer("reshape_shape", np.asarray([1, 8], dtype=np.int64))
    model.set_initializer("mul_mask", np.asarray([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32))
    model.set_initializer("bias", np.asarray([0.25], dtype=np.float32))
    model.set_initializer("res_mask", np.asarray([[0.0, 1.0]], dtype=np.float32))
    model.set_initializer("res_bias", np.asarray([0.0], dtype=np.float32))
    model.set_initializer("bn_scale", np.asarray([1.0], dtype=np.float32))
    model.set_initializer("bn_bias", np.asarray([0.0], dtype=np.float32))
    model.set_initializer("bn_mean", np.asarray([0.0], dtype=np.float32))
    model.set_initializer("bn_var", np.asarray([1.0], dtype=np.float32))
    model.set_initializer("scale1", np.asarray(1.0, dtype=np.float32))

    model.save("symdep_small_lookup_like_chain.onnx")

    ret = model.analysis(symbolic_dependencies)
    symdeps = ret["symbolic_dependencies"]
    tdeps = symdeps["tensor_dependencies"]
    node_local = symdeps["node_local_dependencies"]

    assert tdeps["q0"][0] == {("q0", 0)}
    assert tdeps["q0"][3] == {("q0", 3)}
    assert tdeps["gathered"] == [{("q0", 3)}, {("q0", 1)}]
    assert tdeps["expanded"] == [{("q0", 3)}, {("q0", 1)}, {("q0", 3)}, {("q0", 1)}]
    assert tdeps["tiled"] == [
        {("q0", 3)},
        {("q0", 1)},
        {("q0", 3)},
        {("q0", 1)},
        {("q0", 3)},
        {("q0", 1)},
        {("q0", 3)},
        {("q0", 1)},
    ]
    assert tdeps["reshaped"] == [
        {("q0", 3)},
        {("q0", 1)},
        {("q0", 3)},
        {("q0", 1)},
        {("q0", 3)},
        {("q0", 1)},
        {("q0", 3)},
        {("q0", 1)},
    ]
    assert tdeps["multiplied"] == [
        {("q0", 3)},
        set(),
        {("q0", 3)},
        set(),
        {("q0", 3)},
        set(),
        {("q0", 3)},
        set(),
    ]
    assert tdeps["reduced"] == [{("q0", 3)}]
    assert tdeps["shifted"] == [{("q0", 3)}]
    assert tdeps["relu_out"] == [{("q0", 3)}]
    assert tdeps["res_mul"] == [set(), {("q0", 1)}]
    assert tdeps["res_red"] == [{("q0", 1)}]
    assert tdeps["res_shifted"] == [{("q0", 1)}]
    assert tdeps["merged"] == [{("q0", 3), ("q0", 1)}]
    assert tdeps["bn_out"] == [{("q0", 3), ("q0", 1)}]
    assert tdeps["out"] == [{("out", 0)}]

    assert node_local["0:BipolarQuant"]["q0"] == [
        {("x", 0)},
        {("x", 1)},
        {("x", 2)},
        {("x", 3)},
        {("x", 4)},
        {("x", 5)},
    ]
    assert node_local["1:Gather"]["gathered"] == [{("q0", 3)}, {("q0", 1)}]
    assert node_local["2:Expand"]["expanded"] == [
        {("gathered", 0)},
        {("gathered", 1)},
        {("gathered", 0)},
        {("gathered", 1)},
    ]
    assert node_local["3:Tile"]["tiled"] == [
        {("expanded", 0)},
        {("expanded", 1)},
        {("expanded", 0)},
        {("expanded", 1)},
        {("expanded", 2)},
        {("expanded", 3)},
        {("expanded", 2)},
        {("expanded", 3)},
    ]
    assert node_local["4:Reshape"]["reshaped"] == [
        {("tiled", 0)},
        {("tiled", 1)},
        {("tiled", 2)},
        {("tiled", 3)},
        {("tiled", 4)},
        {("tiled", 5)},
        {("tiled", 6)},
        {("tiled", 7)},
    ]
    assert node_local["5:Mul"]["multiplied"] == [
        {("reshaped", 0)},
        set(),
        {("reshaped", 2)},
        set(),
        {("reshaped", 4)},
        set(),
        {("reshaped", 6)},
        set(),
    ]
    assert node_local["6:ReduceSum"]["reduced"] == [
        {("multiplied", 0), ("multiplied", 1), ("multiplied", 2), ("multiplied", 3), ("multiplied", 4), ("multiplied", 5), ("multiplied", 6), ("multiplied", 7)}
    ]
    assert node_local["7:Add"]["shifted"] == [{("reduced", 0)}]
    assert node_local["8:Relu"]["relu_out"] == [{("shifted", 0)}]
    assert node_local["9:Mul"]["res_mul"] == [set(), {("gathered", 1)}]
    assert node_local["10:ReduceSum"]["res_red"] == [{("res_mul", 0), ("res_mul", 1)}]
    assert node_local["11:Add"]["res_shifted"] == [{("res_red", 0)}]
    assert node_local["12:Add"]["merged"] == [{("relu_out", 0), ("res_shifted", 0)}]
    assert node_local["14:BipolarQuant"]["out"] == [{("bn_out", 0)}]