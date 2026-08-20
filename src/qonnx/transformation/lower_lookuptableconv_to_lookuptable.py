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
from onnx import helper

from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name

LNN_DOMAIN = "qonnx.custom_op.lnn"
GENERAL_DOMAIN = "qonnx.custom_op.general"


class LowerLookupTableConvToLookupTable(Transformation):
    """Replace LookupTableConv nodes (qonnx.custom_op.lnn) with an Im2Col node
    (qonnx.custom_op.general) followed by a chain of LookupTable nodes
    (qonnx.custom_op.lnn), one per tree level.

    This relies on the equivalence documented in
    docs/qonnx-custom-ops/lookuptableconv_v1.md: the leaf level is exactly
    Im2Col + LookupTable over the patch axis, and every level above it combines
    the previous level's output using a fixed, position-independent pairing
    that a plain LookupTable (with a trivial, arithmetically-generated
    `indices`) can express unchanged, since kernel blocks in the flattened
    (M * nodes_per_kernel) axis never straddle a pairing boundary.

    Only 2D LookupTableConv nodes are supported, matching Im2Col's own
    restriction to NHWC 2D inputs.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node in list(graph.node):
            if node.op_type != "LookupTableConv" or node.domain != LNN_DOMAIN:
                continue
            self._lower_node(model, node)
            graph_modified = True
        return (model, graph_modified)

    def _lower_node(self, model, node):
        graph = model.graph
        x_name, idx_name = node.input[0], node.input[1]
        table_name = node.input[2] if len(node.input) == 3 else None

        tree_depth_attr = get_by_name(node.attribute, "tree_depth")
        tree_depth = tree_depth_attr.i if tree_depth_attr is not None else 1
        kernel_shape = list(get_by_name(node.attribute, "kernel_shape").ints)
        strides_attr = get_by_name(node.attribute, "strides")
        strides = list(strides_attr.ints) if strides_attr is not None else [1] * len(kernel_shape)
        pads_attr = get_by_name(node.attribute, "pads")
        pads = list(pads_attr.ints) if pads_attr is not None else [0] * (2 * len(kernel_shape))

        assert len(kernel_shape) == 2, "LowerLookupTableConvToLookupTable only supports 2D LookupTableConv"

        indices = model.get_initializer(idx_name)
        assert indices is not None, "LookupTableConv's indices must be a known initializer"
        m, p, lut_rank = indices.shape

        ishape = model.get_tensor_shape(x_name)
        assert ishape is not None and len(ishape) == 4, "LookupTableConv's input shape must be known and 4D (NHWC)"
        _, ifm_h, ifm_w, ifm_c = ishape
        x_dtype = model.get_tensor_datatype(x_name)
        x_elem_type = model.get_tensor_valueinfo(x_name).type.tensor_type.elem_type

        im2col_out = model.make_new_valueinfo_name()
        im2col_node = helper.make_node(
            "Im2Col",
            [x_name],
            [im2col_out],
            domain=GENERAL_DOMAIN,
            stride=strides,
            kernel_size=kernel_shape,
            pad_amount=pads,
            input_shape="(1,{},{},{})".format(ifm_h, ifm_w, ifm_c),
        )
        # onnxruntime type-checks plain ONNX ops (e.g. Gather) against a real declared
        # elem_type, so every new intermediate tensor needs an actual ValueInfoProto,
        # not just the qonnx-level datatype annotation set below
        graph.value_info.append(helper.make_tensor_value_info(im2col_out, x_elem_type, None))
        model.set_tensor_datatype(im2col_out, x_dtype)
        new_nodes = [im2col_node]

        if tree_depth == 0:
            gather_idx_name = model.make_new_valueinfo_name()
            model.set_initializer(gather_idx_name, indices.reshape(m, lut_rank).astype(np.int64))
            new_nodes.append(helper.make_node("Gather", [im2col_out, gather_idx_name], [node.output[0]], axis=-1))
        else:
            table, table_elem_type = model.get_initializer(table_name, return_dtype=True)
            assert table is not None, "LookupTableConv's table must be a known initializer when tree_depth >= 1"
            np_dtype = table.dtype

            hidden = im2col_out
            row_offset = 0
            for level in range(tree_depth):
                nodes_per_kernel = lut_rank ** (tree_depth - 1 - level)
                total_nodes = m * nodes_per_kernel

                if level == 0:
                    level_indices = indices.reshape(total_nodes, lut_rank)
                else:
                    # canonical, position-independent pairing: node j consumes children
                    # j*lut_rank .. j*lut_rank+lut_rank-1 of the previous level, and kernel
                    # blocks always divide evenly by lut_rank, so no explicit indices are
                    # needed here beyond this arithmetic identity pattern
                    level_indices = (np.arange(total_nodes)[:, None] * lut_rank + np.arange(lut_rank)[None, :])

                level_table = table[:, row_offset : row_offset + nodes_per_kernel, :].reshape(
                    total_nodes, 2**lut_rank
                )
                row_offset += nodes_per_kernel

                idx_name_l = model.make_new_valueinfo_name()
                tab_name_l = model.make_new_valueinfo_name()
                model.set_initializer(idx_name_l, level_indices.astype(np.int64))
                model.set_initializer(tab_name_l, level_table.astype(np_dtype))

                out_name = node.output[0] if level == tree_depth - 1 else model.make_new_valueinfo_name()
                if level < tree_depth - 1:
                    graph.value_info.append(helper.make_tensor_value_info(out_name, table_elem_type, None))
                    model.set_tensor_datatype(out_name, model.get_tensor_datatype(table_name))
                new_nodes.append(
                    helper.make_node(
                        "LookupTable",
                        [hidden, idx_name_l, tab_name_l],
                        [out_name],
                        domain=LNN_DOMAIN,
                        input_bits=1,
                    )
                )
                hidden = out_name

        insert_idx = list(graph.node).index(node)
        for i, new_node in enumerate(new_nodes):
            graph.node.insert(insert_idx + i, new_node)
        graph.node.remove(node)
