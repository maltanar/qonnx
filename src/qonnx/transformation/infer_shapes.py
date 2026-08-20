# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import onnx.shape_inference as si

import qonnx.custom_op.registry as registry
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import is_custom_op
from qonnx.transformation.base import Transformation


def _make_shape_compatible_op(node, model):
    """Return a shape-compatible non-QONNX op for a given QONNX op, or None if this
    node's input shape(s) cannot be resolved yet. Used for shape inference with
    custom ops."""
    assert is_custom_op(node.domain), "Node domain is not a registered custom op domain"
    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.getCustomOp(node)
        return inst.make_shape_compatible_op(model)
    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)


def _hide_finn_ops(model):
    """Replace any QONNX ops by shape-compatible ones, and return a dict that
    can be used to map the string representations of the new (shape-compatible)
    ops back to the old ops. Nodes whose make_shape_compatible_op returns None
    (input shapes not resolvable yet, e.g. produced by another not-yet-hidden
    custom op) are left in place, to be retried on a later InferShapes iteration."""
    hidden_ops = {}
    node_ind = 0
    for node in model.graph.node:
        node_ind += 1
        if is_custom_op(node.domain):
            new_node = _make_shape_compatible_op(node, model)
            if new_node is None:
                continue
            # keep old node name to help debug shape inference issues
            new_node.name = node.name
            hidden_ops[str(new_node)] = node
            model.graph.node.insert(node_ind, new_node)
            model.graph.node.remove(node)
    return hidden_ops


def _restore_finn_ops(model, hidden_ops):
    """Replace any shape-compatible ops with the QONNX ops that originally
    generated them."""
    node_ind = 0
    for node in model.graph.node:
        node_ind += 1
        try:
            old_node = hidden_ops[str(node)]
            model.graph.node.insert(node_ind, old_node)
            model.graph.node.remove(node)
        except KeyError:
            pass


def _tensor_shape_snapshot(model):
    """Serialized ValueInfo for every tensor with a declared shape, used to detect
    when successive rounds of shape inference have stopped making progress."""
    graph = model.graph
    tensors = list(graph.input) + list(graph.output) + list(graph.value_info)
    return {vi.name: vi.SerializeToString() for vi in tensors}


class InferShapes(Transformation):
    """Ensure every tensor in the model has a specified shape (ValueInfo).

    Custom ops are hidden behind shape-compatible standard ONNX ops before calling
    regular ONNX shape inference. A custom op may depend on the (also-hidden) output
    shape of another custom op earlier in the graph, so this is repeated until shapes
    stop changing: each iteration lets shape information propagate one more hop through
    a chain of custom ops, since a custom op with an unresolvable input shape is simply
    left as-is (see _hide_finn_ops) and retried on the next iteration.
    """

    def apply(self, model):
        # bound the number of iterations: a chain of N custom ops needs at most N
        # iterations for shape info to propagate all the way through
        max_iters = len(model.graph.node) + 2
        prev_snapshot = None
        for _ in range(max_iters):
            hidden_ops = _hide_finn_ops(model)
            # call regular ONNX shape inference
            model = ModelWrapper(si.infer_shapes(model.model))
            # bring back hidden ops
            _restore_finn_ops(model, hidden_ops)
            snapshot = _tensor_shape_snapshot(model)
            if snapshot == prev_snapshot:
                break
            prev_snapshot = snapshot
        return (model, False)
