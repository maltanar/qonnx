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

from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp

# maps the ONNX element type of the `table` input to (full container bitwidth, is_signed)
_TABLE_ELEM_TYPE_INFO = {
    TensorProto.BOOL: (1, False),
    TensorProto.UINT8: (8, False),
    TensorProto.INT8: (8, True),
    TensorProto.UINT16: (16, False),
    TensorProto.INT16: (16, True),
    TensorProto.UINT32: (32, False),
    TensorProto.INT32: (32, True),
}


def _get_table_elem_type(model, table_name):
    """Returns the ONNX TensorProto element type of the `table` input."""
    vi = model.get_tensor_valueinfo(table_name)
    if vi is not None:
        return vi.type.tensor_type.elem_type
    _, elem_type = model.get_initializer(table_name, return_dtype=True)
    return elem_type


def lookup_table(x, indices, table, input_bits=1):
    """Numpy reference implementation of the LookupTable op.

    x       : [..., C_in] unsigned integer (or bool) tensor, 0 <= x < 2**input_bits
    indices : [M, K] int, indices[m, k] selects the slot-k input of neuron m
    table   : [M, S] of any integer/bool dtype, S == 2**(K * input_bits)
    returns : [..., M], same dtype as table
    """
    M, K = indices.shape
    S = 2 ** (K * input_bits)
    gathered = x[..., indices].astype(np.int64)  # [..., M, K]
    weight = (2 ** (input_bits * np.arange(K))).astype(np.int64)
    addr = (gathered * weight).sum(-1)  # [..., M]
    dense = table.reshape(M, S)
    flat = addr + np.arange(M, dtype=np.int64) * S
    return dense.reshape(M * S)[flat]  # [..., M], dtype preserved


class LookupTable(CustomOp):
    """Evaluates a set of lookup tables (LUTs) over the last axis of an input
    tensor."""

    def get_nodeattr_types(self):
        return {
            "input_bits": ("i", False, 1),
            "out_bits": ("i", False, 0),
        }

    def make_shape_compatible_op(self, model):
        # oshape is ishape with the last dimension replaced by M
        # where M is the number of rows (neurons) in the indices (idx) input
        node = self.onnx_node
        ishape = model.get_tensor_shape(node.input[0])
        idx_shape = model.get_tensor_shape(node.input[1], fix_missing_init_shape=True)
        # get_tensor_shape returns [] both for a genuinely unknown/unresolved shape
        # (e.g. a pre-declared ValueInfo with only elem_type set) and for a true
        # rank-0 tensor; only the latter is possible here, so treat [] as unresolved
        if ishape is None or not ishape or idx_shape is None or len(idx_shape) != 2:
            # not resolvable yet, e.g. produced by another not-yet-hidden custom op;
            # InferShapes will retry this node on a later iteration
            return None
        m = idx_shape[0]
        oshape = list(ishape[:-1]) + [m]
        # RandomNormal (used by the inherited make_const_shape_op helper) can only produce
        # float tensors, which would corrupt the dtype of table's (usually integer) container
        # once ONNX shape inference runs; use ConstantOfShape instead to preserve it
        elem_type = _get_table_elem_type(model, node.input[2])
        shape_name = node.output[0] + "_shapeop_shape"
        model.set_initializer(shape_name, np.asarray(oshape, dtype=np.int64))
        fill_value = helper.make_tensor(node.output[0] + "_shapeop_value", elem_type, [1], [0])
        return helper.make_node("ConstantOfShape", [shape_name], [node.output[0]], value=fill_value)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        table_name = node.input[2]
        elem_type = _get_table_elem_type(model, table_name)
        assert elem_type in _TABLE_ELEM_TYPE_INFO, "Unsupported table dtype for LookupTable"
        full_bits, signed = _TABLE_ELEM_TYPE_INFO[elem_type]
        out_bits = self.get_nodeattr("out_bits")
        bits = out_bits if out_bits > 0 else full_bits
        if bits == 1 and not signed:
            odt = DataType["BINARY"]
        else:
            odt = DataType["%s%d" % ("INT" if signed else "UINT", bits)]
        model.set_tensor_datatype(node.output[0], odt)

    def execute_node(self, context, graph):
        node = self.onnx_node
        x = context[node.input[0]]
        indices = context[node.input[1]]
        table = context[node.input[2]]
        input_bits = self.get_nodeattr("input_bits")
        context[node.output[0]] = lookup_table(x, indices, table, input_bits=input_bits)

    def verify_node(self):
        info_messages = []

        try:
            self.get_nodeattr("input_bits")
            self.get_nodeattr("out_bits")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("LookupTable needs the following attributes: input_bits, out_bits")

        if len(self.onnx_node.input) == 3:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("LookupTable needs 3 inputs (X, indices, table)")

        return info_messages
