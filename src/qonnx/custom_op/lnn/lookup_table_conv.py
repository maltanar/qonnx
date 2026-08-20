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
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.lnn.lookup_table import _TABLE_ELEM_TYPE_INFO, _get_table_elem_type


def _im2col_patches(x, kernel_shape, strides, pads):
    """x: [N, *D, C] -> patches: [N, *O, prod(kernel_shape) * C], row-major-spatial/channel-minor."""
    n_spatial = len(kernel_shape)
    pad_width = [(0, 0)] + [(pads[d], pads[d + n_spatial]) for d in range(n_spatial)] + [(0, 0)]
    x_padded = np.pad(x, pad_width, mode="constant", constant_values=0)
    out_shape = [
        (x.shape[1 + d] + pads[d] + pads[d + n_spatial] - kernel_shape[d]) // strides[d] + 1 for d in range(n_spatial)
    ]
    patches = np.empty((x.shape[0], *out_shape, np.prod(kernel_shape) * x.shape[-1]), dtype=x.dtype)
    for o in np.ndindex(*out_shape):
        starts = [o[d] * strides[d] for d in range(n_spatial)]
        slices = tuple(slice(s, s + k) for s, k in zip(starts, kernel_shape))
        patch = x_padded[(slice(None), *slices, slice(None))]  # [N, *kernel_shape, C]
        patches[(slice(None), *o)] = patch.reshape(x.shape[0], -1)
    return patches


def lookup_table_conv(x, indices, table, tree_depth, kernel_shape, strides, pads):
    """Numpy reference implementation of the LookupTableConv op.

    x            : [N, *D, C] (channels-last) unsigned integer (or bool) tensor
    indices      : [M, P, lut_rank] int, leaf-level receptive-field connectivity
    table        : [M, N_nodes, 2**lut_rank] of bool/uint8, or None when tree_depth == 0
    tree_depth   : number of tree levels; 0 means passthrough (no lookup evaluated)
    kernel_shape, strides, pads: receptive-field geometry, same convention as ONNX Conv
    returns      : [N, *O, M] (same dtype as table), or [N, *O, M, lut_rank] (same dtype
                   as x) when tree_depth == 0
    """
    lut_rank = indices.shape[-1]
    patches = _im2col_patches(x, kernel_shape, strides, pads)  # [N, *O, C*prod(kernel_shape)]
    gathered = patches[..., indices]  # [N, *O, M, P, lut_rank]

    if tree_depth == 0:
        return gathered[..., 0, :]  # drop the (always size-1) P axis -> [N, *O, M, lut_rank]

    weight = 2 ** np.arange(lut_rank)
    level = gathered  # [..., M, num_nodes_this_level, lut_rank]
    row_offset = 0
    for lvl in range(tree_depth):
        addr = (level * weight).sum(-1)  # [..., M, num_nodes_this_level]
        num_nodes_this_level = addr.shape[-1]
        rows = table[:, row_offset : row_offset + num_nodes_this_level, :]  # [M, num_nodes, 2**lut_rank]
        out = np.take_along_axis(
            np.broadcast_to(rows, (*addr.shape[:-2], *rows.shape)),
            addr[..., None].astype(np.int64),
            axis=-1,
        )[..., 0]  # [..., M, num_nodes_this_level]
        row_offset += num_nodes_this_level
        if lvl < tree_depth - 1:
            level = out.reshape(*out.shape[:-1], num_nodes_this_level // lut_rank, lut_rank)
        else:
            return out[..., 0]  # root: exactly one node per kernel


class LookupTableConv(CustomOp):
    """Evaluates a weight-shared tree of lookup tables (LUTs) over sliding
    receptive fields of a channels-last spatial input tensor."""

    def get_nodeattr_types(self):
        return {
            "tree_depth": ("i", False, 1),
            "kernel_shape": ("ints", True, []),
            "strides": ("ints", False, []),
            "pads": ("ints", False, []),
            "channel_group_size": ("i", False, 0),
        }

    def _geometry(self):
        kernel_shape = self.get_nodeattr("kernel_shape")
        n_spatial = len(kernel_shape)
        strides = self.get_nodeattr("strides") or [1] * n_spatial
        pads = self.get_nodeattr("pads") or [0] * (2 * n_spatial)
        return kernel_shape, strides, pads

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        ishape = model.get_tensor_shape(node.input[0])
        idx_shape = model.get_tensor_shape(node.input[1], fix_missing_init_shape=True)
        if ishape is None or idx_shape is None or len(idx_shape) != 3:
            # not resolvable yet, e.g. produced by another not-yet-hidden custom op;
            # InferShapes will retry this node on a later iteration
            return None
        kernel_shape, strides, pads = self._geometry()
        n_spatial = len(kernel_shape)
        ofm_dims = [
            compute_conv_output_dim(ishape[1 + d], kernel_shape[d], strides[d], pads[d] + pads[d + n_spatial])
            for d in range(n_spatial)
        ]
        m, lut_rank = idx_shape[0], idx_shape[-1]
        tree_depth = self.get_nodeattr("tree_depth")

        if tree_depth == 0:
            oshape = [ishape[0]] + ofm_dims + [m, lut_rank]
            elem_type = _get_table_elem_type(model, node.input[0])
        else:
            oshape = [ishape[0]] + ofm_dims + [m]
            elem_type = _get_table_elem_type(model, node.input[2])

        # RandomNormal (used by the inherited make_const_shape_op helper) can only produce
        # float tensors, which would corrupt the dtype of table's (usually integer) container
        # once ONNX shape inference runs; use ConstantOfShape instead to preserve it
        shape_name = node.output[0] + "_shapeop_shape"
        model.set_initializer(shape_name, np.asarray(oshape, dtype=np.int64))
        fill_value = helper.make_tensor(node.output[0] + "_shapeop_value", elem_type, [1], [0])
        return helper.make_node("ConstantOfShape", [shape_name], [node.output[0]], value=fill_value)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        tree_depth = self.get_nodeattr("tree_depth")
        if tree_depth == 0:
            model.set_tensor_datatype(node.output[0], model.get_tensor_datatype(node.input[0]))
            return

        elem_type = _get_table_elem_type(model, node.input[2])
        assert elem_type in _TABLE_ELEM_TYPE_INFO, "Unsupported table dtype for LookupTableConv"
        bits, signed = _TABLE_ELEM_TYPE_INFO[elem_type]
        if bits == 1 and not signed:
            odt = DataType["BINARY"]
        else:
            odt = DataType["%s%d" % ("INT" if signed else "UINT", bits)]
        model.set_tensor_datatype(node.output[0], odt)

    def execute_node(self, context, graph):
        node = self.onnx_node
        x = context[node.input[0]]
        indices = context[node.input[1]]
        tree_depth = self.get_nodeattr("tree_depth")
        table = context[node.input[2]] if len(node.input) == 3 else None
        kernel_shape, strides, pads = self._geometry()
        context[node.output[0]] = lookup_table_conv(x, indices, table, tree_depth, kernel_shape, strides, pads)

    def verify_node(self):
        info_messages = []

        try:
            self.get_nodeattr("tree_depth")
            self.get_nodeattr("kernel_shape")
            self.get_nodeattr("strides")
            self.get_nodeattr("pads")
            self.get_nodeattr("channel_group_size")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                "LookupTableConv needs the following attributes: "
                "tree_depth, kernel_shape, strides, pads, channel_group_size"
            )

        tree_depth = self.get_nodeattr("tree_depth")
        n_inputs = len(self.onnx_node.input)
        if tree_depth == 0:
            if n_inputs in (2, 3):
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append("LookupTableConv needs 2 or 3 inputs (X, indices[, table]) when tree_depth == 0")
        elif n_inputs == 3:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("LookupTableConv needs 3 inputs (X, indices, table) when tree_depth >= 1")

        return info_messages
