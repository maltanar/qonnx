"""Shared helpers for building small ONNX models around LookupTable in tests."""

import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper

from lnn_spec_proto import DOMAIN, OPSET, make_lookup_table_function


def tensor(name, shape, dtype):
    return helper.make_tensor_value_info(name, dtype, shape)


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def lut_node(inputs, outputs, input_bits=1, out_bits=0, name=None):
    return helper.make_node(
        "LookupTable",
        inputs,
        outputs,
        domain=DOMAIN,
        name=name,
        input_bits=input_bits,
        out_bits=out_bits,
    )


def build_model(nodes, inputs, outputs, initializers, with_function, name):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=initializers)
    functions = [make_lookup_table_function()] if with_function else []
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid(DOMAIN, 1)],
        functions=functions,
    )
    if any(n.domain == "qonnx.custom_op.general" for n in nodes):
        model.opset_import.append(helper.make_opsetid("qonnx.custom_op.general", 1))
    return model


def run_ort(model, feed):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


def random_indices(rng, M, K, C_in):
    return np.stack([rng.choice(C_in, size=K, replace=False) for _ in range(M)]).astype(np.int64)


def im2col_nhwc(x, k):
    """Reference im2col matching qonnx's Im2Col (NHWC, stride 1, no padding)."""
    N, H, W, C = x.shape
    oh, ow = H - k + 1, W - k + 1
    out = np.zeros((N, oh, ow, k * k * C), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            out[:, i, j, :] = x[:, i : i + k, j : j + k, :].reshape(N, -1)
    return out
