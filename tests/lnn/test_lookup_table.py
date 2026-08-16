"""Tests for the proposed LookupTable (LNN) custom op.

Each test builds a small ONNX model around LookupTable, saves and reloads it
(to exercise serialization), executes it through onnxruntime where the model
is fully standard-op decomposable, and checks the result against the numpy
reference implementation in lnn_spec_proto.py.

See docs/qonnx-custom-ops/lookuptable_v1.md for the full specification.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from lnn_spec_proto import lookup_table
from onnx_builders import build_model, im2col_nhwc, init, lut_node, random_indices, run_ort, tensor


def save_and_run(model, feed, tmp_path, fname="model.onnx"):
    path = tmp_path / fname
    onnx.save(model, str(path))
    return run_ort(onnx.load(str(path)), feed)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_gatenet(tmp_path):
    """Two-layer logic gate network: {XOR, AND, OR} feeding a majority gate.

    Chains two LookupTable nodes with no Cast in between, since tab1's dtype
    (uint8) is already one of X's allowed types.
    """
    idx1 = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)  # 3 neurons, fan-in 2
    tab1 = np.array(
        [
            [0, 1, 1, 0],  # XOR
            [0, 0, 0, 1],  # AND
            [0, 1, 1, 1],  # OR
        ],
        dtype=np.uint8,
    )
    idx2 = np.array([[0, 1, 2]], dtype=np.int64)  # 1 neuron, fan-in 3
    tab2 = np.array([[0, 0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)  # majority

    nodes = [
        lut_node(["X", "idx1", "tab1"], ["h"], name="gates"),
        lut_node(["h", "idx2", "tab2"], ["Y"], name="majority"),
    ]
    model = build_model(
        nodes,
        [tensor("X", [None, 4], TensorProto.BOOL)],
        [tensor("Y", [None, 1], TensorProto.UINT8)],
        [init("idx1", idx1), init("tab1", tab1), init("idx2", idx2), init("tab2", tab2)],
        with_function=True,
        name="lnn_gatenet",
    )
    x = np.array([[a, b, c, d] for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)], dtype=bool)
    h = lookup_table(x, idx1, tab1)
    expected = lookup_table(h, idx2, tab2)

    got = save_and_run(model, {"X": x}, tmp_path)
    assert np.array_equal(got, expected)


def test_sparse_layer(rng, tmp_path):
    """Sparse fully-connected LNN layer, fan-in 6 over a 64-element input."""
    C_in, M, K = 64, 32, 6
    idx = random_indices(rng, M, K, C_in)
    tab = rng.integers(0, 2, size=(M, 2**K)).astype(np.uint8)
    x = rng.integers(0, 2, size=(4, C_in)).astype(np.uint8)
    expected = lookup_table(x, idx, tab)

    model = build_model(
        [lut_node(["X", "idx", "tab"], ["Y"], name="lut")],
        [tensor("X", [None, C_in], TensorProto.UINT8)],
        [tensor("Y", [None, M], TensorProto.UINT8)],
        [init("idx", idx), init("tab", tab)],
        with_function=True,
        name="lnn_sparse_layer",
    )
    got = save_and_run(model, {"X": x}, tmp_path)
    assert np.array_equal(got, expected)


def test_conv(rng, tmp_path):
    """Convolutional LNN: Im2Col (NHWC) followed by LookupTable over the patch axis."""
    N, H, W, C, k, M = 1, 16, 16, 8, 3, 16
    patch = k * k * C
    idx = random_indices(rng, M, 4, patch)
    tab = rng.integers(0, 2, size=(M, 2**4)).astype(np.uint8)

    x = rng.integers(0, 2, size=(N, H, W, C)).astype(np.uint8)
    patches = im2col_nhwc(x, k)
    expected = lookup_table(patches, idx, tab)

    # only the LookupTable part is standard-op decomposable and runnable via
    # onnxruntime; Im2Col is a qonnx custom op that has no ORT implementation
    model = build_model(
        [lut_node(["P", "idx", "tab"], ["Y"], name="lut")],
        [tensor("P", [N, H - k + 1, W - k + 1, patch], TensorProto.UINT8)],
        [tensor("Y", [N, H - k + 1, W - k + 1, M], TensorProto.UINT8)],
        [init("idx", idx), init("tab", tab)],
        with_function=True,
        name="lnn_conv_lut_only",
    )
    got = save_and_run(model, {"P": patches}, tmp_path)
    assert np.array_equal(got, expected)


def test_conv_graph_with_im2col_builds(rng):
    """The full conv model (Im2Col + LookupTable) is at least constructible and checkable."""
    N, H, W, C, k, M = 1, 16, 16, 8, 3, 16
    patch = k * k * C
    idx = random_indices(rng, M, 4, patch)
    tab = rng.integers(0, 2, size=(M, 2**4)).astype(np.uint8)

    im2col = onnx.helper.make_node(
        "Im2Col",
        ["X"],
        ["patches"],
        domain="qonnx.custom_op.general",
        stride=[1, 1],
        kernel_size=[k, k],
        input_shape=str((N, H, W, C)),
        pad_amount=[0, 0, 0, 0],
    )
    model = build_model(
        [im2col, lut_node(["patches", "idx", "tab"], ["Y"], name="lut")],
        [tensor("X", [N, H, W, C], TensorProto.UINT8)],
        [tensor("Y", [N, H - k + 1, W - k + 1, M], TensorProto.UINT8)],
        [init("idx", idx), init("tab", tab)],
        with_function=False,
        name="lnn_conv",
    )
    onnx.checker.check_model(model)


def test_hybrid_quant_graph_with_quant_builds():
    """The full model (Quant -> Cast -> LookupTable -> Cast -> Mul) is at least constructible and checkable.

    Quant is a qonnx custom op with no onnxruntime kernel/function, so the full graph can't be executed
    by onnxruntime; see test_hybrid_quant_lut_part below for the executable portion.
    """
    rng = np.random.default_rng(0)
    C_in, M, K, b = 8, 4, 3, 2
    scale = np.float32(0.25)
    idx = random_indices(rng, M, K, C_in)
    tab = rng.integers(0, 4, size=(M, 2 ** (K * b))).astype(np.uint8)
    out_scale = np.float32(0.5)

    nodes = [
        onnx.helper.make_node(
            "Quant",
            ["X", "scale", "zeropt", "bitwidth"],
            ["Xq"],
            domain="qonnx.custom_op.general",
            signed=0,
            narrow=0,
            rounding_mode="ROUND",
        ),
        onnx.helper.make_node("Div", ["Xq", "scale"], ["Xint"]),  # float -> integer levels
        onnx.helper.make_node("Cast", ["Xint"], ["Xu"], to=TensorProto.UINT8),  # into LookupTable's integer domain
        lut_node(["Xu", "idx", "tab"], ["Ylut"], input_bits=b, out_bits=2, name="lut"),  # 2 of 8 uint8 bits used
        onnx.helper.make_node("Cast", ["Ylut"], ["Ylutf"], to=TensorProto.FLOAT),  # bit string -> float32
        onnx.helper.make_node("Mul", ["Ylutf", "out_scale"], ["Y"]),
    ]
    model = build_model(
        nodes,
        [tensor("X", [None, C_in], TensorProto.FLOAT)],
        [tensor("Y", [None, M], TensorProto.FLOAT)],
        [
            init("scale", scale),
            init("zeropt", np.float32(0.0)),
            init("bitwidth", np.float32(b)),
            init("idx", idx),
            init("tab", tab),
            init("out_scale", out_scale),
        ],
        with_function=True,
        name="lnn_hybrid_quant",
    )
    onnx.checker.check_model(model)


def test_hybrid_quant_lut_part(tmp_path):
    """Cast into LookupTable's integer domain -> Cast+Mul for a scaled output, executed via onnxruntime."""
    rng = np.random.default_rng(0)
    C_in, M, K, b = 8, 4, 3, 2
    scale = np.float32(0.25)
    idx = random_indices(rng, M, K, C_in)
    tab = rng.integers(0, 4, size=(M, 2 ** (K * b))).astype(np.uint8)  # 6:2 LUTs, 2-bit output packed in uint8
    out_scale = np.float32(0.5)  # scaled activations: table levels 0, 0.5, 1.0, 1.5

    nodes = [
        onnx.helper.make_node("Cast", ["Xint"], ["Xu"], to=TensorProto.UINT8),  # into LookupTable's integer domain
        lut_node(["Xu", "idx", "tab"], ["Ylut"], input_bits=b, out_bits=2, name="lut"),  # 2 of 8 uint8 bits used
        onnx.helper.make_node("Cast", ["Ylut"], ["Ylutf"], to=TensorProto.FLOAT),  # bit string -> float32
        onnx.helper.make_node("Mul", ["Ylutf", "out_scale"], ["Y"]),
    ]
    model = build_model(
        nodes,
        [tensor("Xint", [None, C_in], TensorProto.FLOAT)],
        [tensor("Y", [None, M], TensorProto.FLOAT)],
        [init("idx", idx), init("tab", tab), init("out_scale", out_scale)],
        with_function=True,
        name="lnn_hybrid_quant_lut_part",
    )

    xf = rng.uniform(0, 0.75, size=(4, C_in)).astype(np.float32)
    xint = np.rint(xf / scale).astype(np.float32)  # what Quant -> Div would have produced
    expected = lookup_table(xint.astype(np.uint8), idx, tab, input_bits=b).astype(np.float32) * out_scale

    got = save_and_run(model, {"Xint": xint}, tmp_path)
    assert np.allclose(got, expected)

