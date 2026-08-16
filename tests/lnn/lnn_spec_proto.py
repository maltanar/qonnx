"""Prototype of the proposed LookupTable (LNN) custom op for QONNX.

X is restricted to unsigned integer types (uint8/uint16/uint32/bool), so the
whole address computation is pure integer arithmetic -- no rounding, and
no Cast-to-float32 needed to chain two LookupTable nodes together when
the upstream table's dtype already matches one of X's allowed types.

Fixed fan-in K per node (inferred from indices.shape[1]). Table has shape
[M, S] and any ONNX integer/bool dtype; that dtype is also the output
dtype, so a multi-bit output is just the raw table entry -- no separate
"output bits" axis. Contains a numpy reference implementation and a
builder for the equivalent ONNX function body, built purely from
standard ONNX ops, so the custom node stays opaque at the graph level
but is still executable by any runtime.

See docs/qonnx-custom-ops/lookuptable_v1.md for the full specification.
"""

import numpy as np
from onnx import AttributeProto, TensorProto, helper, numpy_helper

DOMAIN = "qonnx.custom_op.lnn"
OPSET = 21

# ---------------------------------------------------------------------------
# numpy reference implementation
# ---------------------------------------------------------------------------


def lut_address(x, indices, input_bits):
    """Compute the LUT address for every neuron.

    x       : [..., C_in] unsigned integer tensor, 0 <= x < 2**input_bits
    indices : [M, K] int
    returns : [..., M] int64
    """
    gathered = x[..., indices].astype(np.int64)  # [..., M, K]
    K = indices.shape[1]
    weight = (2 ** (input_bits * np.arange(K))).astype(np.int64)
    return (gathered * weight).sum(-1)


def lookup_table(x, indices, table, input_bits=1):
    """Reference execution of the proposed LookupTable op.

    table   : [M, S] of any integer/bool dtype; that dtype is also Y's dtype.
    returns : [..., M], same dtype as table
    """
    M, K = indices.shape
    S = 2 ** (K * input_bits)
    dense = table.reshape(M, S)
    addr = lut_address(x, indices, input_bits)  # [..., M]
    flat = addr + np.arange(M, dtype=np.int64) * S
    return dense.reshape(M * S)[flat]  # [..., M], dtype preserved


# ---------------------------------------------------------------------------
# ONNX function body, standard ops only
# ---------------------------------------------------------------------------


def _const(name, arr):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.asarray(arr), name + "_v"))


def make_lookup_table_function():
    """FunctionProto for LookupTable(X, indices, table) -> Y."""
    n = []
    ib_node = helper.make_node("Constant", [], ["ib"])
    ib_node.attribute.append(AttributeProto(name="value_int", type=AttributeProto.INT, ref_attr_name="input_bits"))
    n.append(ib_node)
    n.append(_const("i0", np.array([0], dtype=np.int64)))
    n.append(_const("i1", np.array([1], dtype=np.int64)))
    n.append(_const("s0", np.array(0, dtype=np.int64)))
    n.append(_const("s1", np.array(1, dtype=np.int64)))
    n.append(_const("two", np.array(2, dtype=np.int64)))
    n.append(_const("m1", np.array([-1], dtype=np.int64)))

    # shapes: indices [M, K], table [M, S]
    n.append(helper.make_node("Shape", ["indices"], ["ishape"]))
    n.append(helper.make_node("Gather", ["ishape", "i0"], ["M1"]))  # [1]
    n.append(helper.make_node("Gather", ["ishape", "i1"], ["K1"]))  # [1]
    n.append(helper.make_node("Shape", ["table"], ["tshape"]))
    n.append(helper.make_node("Gather", ["tshape", "i1"], ["S1"]))  # [1]
    n.append(helper.make_node("Squeeze", ["K1", "i0"], ["Ks"]))
    n.append(helper.make_node("Squeeze", ["M1", "i0"], ["Ms"]))

    # per-slot address weights 2**(input_bits*k), computed as plain int64 (Pow
    # supports integer base/exponent), no float round-trip needed
    n.append(helper.make_node("Range", ["s0", "Ks", "s1"], ["krange"]))
    n.append(helper.make_node("Mul", ["krange", "ib"], ["kexp"]))
    n.append(helper.make_node("Pow", ["two", "kexp"], ["wt"]))

    # gather the fan-in values (X's dtype) and reduce to an int64 address
    n.append(helper.make_node("Gather", ["X", "indices"], ["fanin"], axis=-1))
    n.append(helper.make_node("Cast", ["fanin"], ["faninI"], to=TensorProto.INT64))
    n.append(helper.make_node("Mul", ["faninI", "wt"], ["weighted"]))
    n.append(helper.make_node("ReduceSum", ["weighted", "m1"], ["addr"], keepdims=0))

    # flatten (neuron, address) into a single gather over the fully-flattened table;
    # the result already has shape [..., M], matching indices' shape, so no reshape is needed
    n.append(helper.make_node("Range", ["s0", "Ms", "s1"], ["mrange"]))
    n.append(helper.make_node("Mul", ["mrange", "S1"], ["moff"]))
    n.append(helper.make_node("Add", ["addr", "moff"], ["flat_addr"]))
    n.append(helper.make_node("Reshape", ["table", "m1"], ["tbl1d"]))
    n.append(helper.make_node("Gather", ["tbl1d", "flat_addr"], ["Y"], axis=0))

    return helper.make_function(
        domain=DOMAIN,
        fname="LookupTable",
        inputs=["X", "indices", "table"],
        outputs=["Y"],
        nodes=n,
        opset_imports=[helper.make_opsetid("", OPSET)],
        attributes=["input_bits"],
    )
