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

from collections.abc import Iterable
from typing import Any

import numpy as np

from qonnx.util.basic import get_by_name

# Each dependency source is represented by (tensor_name, flat_element_index).
SourceElem = tuple[str, int]
ElemDeps = set[SourceElem]
TensorDeps = list[ElemDeps]

_QUANTIZER_OPS = {"Quant", "BipolarQuant"}


def _numel(shape: list[int] | tuple[int, ...] | None) -> int:
    if shape is None or len(shape) == 0:
        return 0
    return int(np.prod(shape))


def _clone_tensor_deps(td: TensorDeps) -> TensorDeps:
    return [set(x) for x in td]


def _empty_tensor_deps(shape: list[int] | tuple[int, ...] | None) -> TensorDeps:
    return [set() for _ in range(_numel(shape))]


def _identity_tensor_deps(tensor_name: str, shape: list[int] | tuple[int, ...] | None) -> TensorDeps:
    return [{(tensor_name, i)} for i in range(_numel(shape))]


def _union_deps(dep_iter: Iterable[ElemDeps]) -> ElemDeps:
    ret: ElemDeps = set()
    for dep in dep_iter:
        ret |= dep
    return ret


def _get_attr(node, attr_name: str, default: Any = None) -> Any:
    attr = get_by_name(node.attribute, attr_name)
    if attr is None:
        return default
    if attr.HasField("i"):
        return int(attr.i)
    if attr.HasField("f"):
        return float(attr.f)
    if attr.HasField("s"):
        return attr.s
    if len(attr.ints) > 0:
        return [int(x) for x in attr.ints]
    if len(attr.floats) > 0:
        return [float(x) for x in attr.floats]
    return default


def _normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def _broadcast_input_coord(out_coord: tuple[int, ...], in_shape: list[int]) -> tuple[int, ...]:
    out_rank = len(out_coord)
    in_rank = len(in_shape)
    ret = []
    for i in range(in_rank):
        out_dim_ind = out_rank - in_rank + i
        out_dim_val = out_coord[out_dim_ind]
        in_dim = in_shape[i]
        ret.append(0 if in_dim == 1 else out_dim_val)
    return tuple(ret)


def _lookup_dep(td: TensorDeps, shape: list[int], coord: tuple[int, ...]) -> ElemDeps:
    if len(shape) == 0:
        return set()
    idx = int(np.ravel_multi_index(coord, shape))
    if idx < 0 or idx >= len(td):
        return set()
    return set(td[idx])


def _build_output_with_broadcast(inputs: list[tuple[TensorDeps, list[int]]], out_shape: list[int]) -> TensorDeps:
    out_numel = _numel(out_shape)
    if out_numel == 0:
        return []
    out_td = []
    for flat_ind in range(out_numel):
        out_coord = tuple(np.unravel_index(flat_ind, out_shape))
        deps = set()
        for in_td, in_shape in inputs:
            if len(in_shape) == 0:
                continue
            in_coord = _broadcast_input_coord(out_coord, in_shape)
            deps |= _lookup_dep(in_td, in_shape, in_coord)
        out_td.append(deps)
    return out_td


def _map_elementwise_passthrough(in_td: TensorDeps, in_shape: list[int], out_shape: list[int]) -> TensorDeps:
    in_numel = _numel(in_shape)
    out_numel = _numel(out_shape)
    if in_numel == out_numel:
        return _clone_tensor_deps(in_td)
    if in_numel == 0 or out_numel == 0:
        return _empty_tensor_deps(out_shape)
    # Fallback to broadcast-style map if only one side differs but is compatible.
    return _build_output_with_broadcast([(in_td, in_shape)], out_shape)


def _slice_ranges(data_shape: list[int], starts, ends, axes, steps) -> list[np.ndarray]:
    rank = len(data_shape)
    full_axes = list(range(rank)) if axes is None else list(axes)
    if steps is None:
        steps = [1] * len(full_axes)
    ranges = [np.arange(dim, dtype=np.int64) for dim in data_shape]
    for i, ax in enumerate(full_axes):
        axn = _normalize_axis(int(ax), rank)
        st = int(starts[i])
        en = int(ends[i])
        sp = int(steps[i])
        rng = range(data_shape[axn])[slice(st, en, sp)]
        ranges[axn] = np.asarray(list(rng), dtype=np.int64)
    return ranges


def _is_constant_input(model, tensor_name: str) -> bool:
    return model.get_initializer(tensor_name) is not None


def _get_input_shape(model, tensor_name: str) -> list[int]:
    shp = model.get_tensor_shape(tensor_name)
    assert shp is not None, f"symbolic_dependencies requires resolved shape for tensor '{tensor_name}'"
    return shp


def _fallback_union_handler(model, node, tensor_deps, include_constants):
    out_shapes = [_get_input_shape(model, o) for o in node.output]
    dyn_inputs = []
    for i_name in node.input:
        if i_name == "":
            continue
        if (not include_constants) and _is_constant_input(model, i_name):
            continue
        dyn_inputs.append(i_name)
    union = _union_deps([_union_deps(tensor_deps.get(i_name, [])) for i_name in dyn_inputs])
    ret = {}
    for o_name, o_shape in zip(node.output, out_shapes):
        ret[o_name] = [set(union) for _ in range(_numel(o_shape))]
    return ret


def _local_symbolic_handler(model, node, tensor_deps, include_constants=False, prune_zero_weights=True):
    op_type = node.op_type
    in_names = [x for x in node.input if x != ""]
    out_names = list(node.output)

    in_shapes = {n: _get_input_shape(model, n) for n in in_names}
    out_shapes = {n: _get_input_shape(model, n) for n in out_names}

    in_dyn = []
    for n in in_names:
        if include_constants or (not _is_constant_input(model, n)):
            in_dyn.append(n)

    if op_type in ["Identity", "Relu", "BatchNormalization", "Expand"]:
        in0 = in_names[0]
        return {
            out_names[0]: _map_elementwise_passthrough(
                tensor_deps.get(in0, _empty_tensor_deps(in_shapes[in0])),
                in_shapes[in0],
                out_shapes[out_names[0]],
            )
        }

    if op_type in ["Reshape", "Flatten", "Squeeze", "Unsqueeze", "Quant", "BipolarQuant"]:
        in0 = in_names[0]
        return {
            out_names[0]: _map_elementwise_passthrough(
                tensor_deps.get(in0, _empty_tensor_deps(in_shapes[in0])),
                in_shapes[in0],
                out_shapes[out_names[0]],
            )
        }

    if op_type == "Transpose":
        in0 = in_names[0]
        in0_shape = in_shapes[in0]
        out0_shape = out_shapes[out_names[0]]
        perm = _get_attr(node, "perm", list(range(len(in0_shape) - 1, -1, -1)))
        in_td = tensor_deps.get(in0, _empty_tensor_deps(in0_shape))
        out_td = _empty_tensor_deps(out0_shape)
        for out_flat_ind in range(_numel(out0_shape)):
            out_coord = tuple(np.unravel_index(out_flat_ind, out0_shape))
            in_coord = tuple(out_coord[perm.index(i)] for i in range(len(perm)))
            out_td[out_flat_ind] = _lookup_dep(in_td, in0_shape, in_coord)
        return {out_names[0]: out_td}

    if op_type in ["Add", "Mul"]:
        out0_shape = out_shapes[out_names[0]]
        inputs = []
        for n in in_dyn:
            inputs.append((tensor_deps.get(n, _empty_tensor_deps(in_shapes[n])), in_shapes[n]))
        out_td = _build_output_with_broadcast(inputs, out0_shape)

        if prune_zero_weights and op_type == "Mul" and len(in_names) == 2:
            a_name, b_name = in_names
            a_init = model.get_initializer(a_name)
            b_init = model.get_initializer(b_name)
            if a_init is not None or b_init is not None:
                out_numel = _numel(out0_shape)
                a_td = tensor_deps.get(a_name, _empty_tensor_deps(in_shapes[a_name]))
                b_td = tensor_deps.get(b_name, _empty_tensor_deps(in_shapes[b_name]))
                out_td = []
                for out_flat_ind in range(out_numel):
                    out_coord = tuple(np.unravel_index(out_flat_ind, out0_shape))
                    deps = set()
                    if a_init is not None:
                        b_coord = _broadcast_input_coord(out_coord, in_shapes[b_name])
                        if float(np.asarray(a_init)[_broadcast_input_coord(out_coord, in_shapes[a_name])]) != 0.0:
                            deps |= _lookup_dep(b_td, in_shapes[b_name], b_coord)
                    if b_init is not None:
                        a_coord = _broadcast_input_coord(out_coord, in_shapes[a_name])
                        if float(np.asarray(b_init)[_broadcast_input_coord(out_coord, in_shapes[b_name])]) != 0.0:
                            deps |= _lookup_dep(a_td, in_shapes[a_name], a_coord)
                    if a_init is None and b_init is None:
                        deps |= _lookup_dep(a_td, in_shapes[a_name], _broadcast_input_coord(out_coord, in_shapes[a_name]))
                        deps |= _lookup_dep(b_td, in_shapes[b_name], _broadcast_input_coord(out_coord, in_shapes[b_name]))
                    out_td.append(deps)

        return {out_names[0]: out_td}

    if op_type == "Concat":
        axis = _normalize_axis(int(_get_attr(node, "axis", 0)), len(out_shapes[out_names[0]]))
        out_shape = out_shapes[out_names[0]]
        out_td = _empty_tensor_deps(out_shape)
        offsets = []
        cur = 0
        for n in in_names:
            offsets.append(cur)
            cur += in_shapes[n][axis]
        for out_flat_ind in range(_numel(out_shape)):
            out_coord = list(np.unravel_index(out_flat_ind, out_shape))
            axv = out_coord[axis]
            sel = None
            for i, n in enumerate(in_names):
                off = offsets[i]
                dim = in_shapes[n][axis]
                if off <= axv < off + dim:
                    sel = (n, off)
                    break
            if sel is None:
                continue
            in_name, off = sel
            in_coord = list(out_coord)
            in_coord[axis] = in_coord[axis] - off
            in_td = tensor_deps.get(in_name, _empty_tensor_deps(in_shapes[in_name]))
            out_td[out_flat_ind] = _lookup_dep(in_td, in_shapes[in_name], tuple(in_coord))
        return {out_names[0]: out_td}

    if op_type == "Tile":
        in0 = in_names[0]
        in0_shape = in_shapes[in0]
        out0_shape = out_shapes[out_names[0]]
        in_td = tensor_deps.get(in0, _empty_tensor_deps(in0_shape))
        out_td = _empty_tensor_deps(out0_shape)
        for out_flat_ind in range(_numel(out0_shape)):
            out_coord = tuple(np.unravel_index(out_flat_ind, out0_shape))
            in_coord = tuple(out_coord[i] % in0_shape[i] for i in range(len(in0_shape)))
            out_td[out_flat_ind] = _lookup_dep(in_td, in0_shape, in_coord)
        return {out_names[0]: out_td}

    if op_type == "ReduceSum":
        in0 = in_names[0]
        in0_shape = in_shapes[in0]
        out0_shape = out_shapes[out_names[0]]
        in_td = tensor_deps.get(in0, _empty_tensor_deps(in0_shape))
        rank = len(in0_shape)
        axes = _get_attr(node, "axes", None)
        noop_with_empty_axes = int(_get_attr(node, "noop_with_empty_axes", 0))
        if axes is None and len(in_names) > 1:
            axes_init = model.get_initializer(in_names[1])
            if axes_init is not None:
                axes = [int(x) for x in np.asarray(axes_init).flatten()]
        if axes is None:
            if noop_with_empty_axes == 1:
                return {
                    out_names[0]: _map_elementwise_passthrough(
                        in_td,
                        in0_shape,
                        out0_shape,
                    )
                }
            red_axes = set(range(rank))
        else:
            red_axes = set(_normalize_axis(int(a), rank) for a in axes)
        keepdims = int(_get_attr(node, "keepdims", 1))

        out_td = _empty_tensor_deps(out0_shape)
        for out_flat_ind in range(_numel(out0_shape)):
            out_coord = tuple(np.unravel_index(out_flat_ind, out0_shape))
            if keepdims == 1:
                base_coord = list(out_coord)
            else:
                base_coord = []
                oi = 0
                for ax in range(rank):
                    if ax in red_axes:
                        base_coord.append(0)
                    else:
                        base_coord.append(out_coord[oi])
                        oi += 1
            deps = set()
            red_axes_sorted = sorted(list(red_axes))
            red_ranges = [range(in0_shape[ax]) for ax in red_axes_sorted]
            if len(red_axes_sorted) == 0:
                deps |= _lookup_dep(in_td, in0_shape, tuple(base_coord))
            else:
                for red_coords in np.ndindex(*[len(r) for r in red_ranges]):
                    coord = list(base_coord)
                    for j, ax in enumerate(red_axes_sorted):
                        coord[ax] = red_ranges[j][red_coords[j]]
                    deps |= _lookup_dep(in_td, in0_shape, tuple(coord))
            out_td[out_flat_ind] = deps
        return {out_names[0]: out_td}

    if op_type == "Slice":
        in0 = in_names[0]
        in0_shape = in_shapes[in0]
        out0_shape = out_shapes[out_names[0]]
        in_td = tensor_deps.get(in0, _empty_tensor_deps(in0_shape))

        starts = model.get_initializer(in_names[1]) if len(in_names) > 1 else None
        ends = model.get_initializer(in_names[2]) if len(in_names) > 2 else None
        axes = model.get_initializer(in_names[3]) if len(in_names) > 3 else None
        steps = model.get_initializer(in_names[4]) if len(in_names) > 4 else None

        if starts is None or ends is None:
            return _fallback_union_handler(model, node, tensor_deps, include_constants)

        ranges = _slice_ranges(
            in0_shape,
            np.asarray(starts).flatten(),
            np.asarray(ends).flatten(),
            None if axes is None else np.asarray(axes).flatten(),
            None if steps is None else np.asarray(steps).flatten(),
        )
        out_td = _empty_tensor_deps(out0_shape)
        for out_flat_ind in range(_numel(out0_shape)):
            out_coord = tuple(np.unravel_index(out_flat_ind, out0_shape))
            in_coord = tuple(int(ranges[d][out_coord[d]]) for d in range(len(out0_shape)))
            out_td[out_flat_ind] = _lookup_dep(in_td, in0_shape, in_coord)
        return {out_names[0]: out_td}

    if op_type == "Gather":
        data_name = in_names[0]
        idx_name = in_names[1] if len(in_names) > 1 else None
        data_shape = in_shapes[data_name]
        out_shape = out_shapes[out_names[0]]
        data_td = tensor_deps.get(data_name, _empty_tensor_deps(data_shape))
        axis = _normalize_axis(int(_get_attr(node, "axis", 0)), len(data_shape))
        if idx_name is None or model.get_initializer(idx_name) is None:
            return _fallback_union_handler(model, node, tensor_deps, include_constants)
        indices = np.asarray(model.get_initializer(idx_name), dtype=np.int64)

        out_td = _empty_tensor_deps(out_shape)
        out_numel = _numel(out_shape)
        indices_shape = list(indices.shape)
        for out_flat_ind in range(out_numel):
            out_coord = tuple(np.unravel_index(out_flat_ind, out_shape))
            pre_rank = axis
            idx_rank = len(indices_shape)
            data_rank = len(data_shape)
            idx_coord = out_coord[pre_rank : pre_rank + idx_rank]
            gather_idx = int(indices[idx_coord])
            if gather_idx < 0:
                gather_idx += data_shape[axis]
            in_coord = []
            out_ptr = 0
            for ax in range(data_rank):
                if ax == axis:
                    in_coord.append(gather_idx)
                else:
                    in_coord.append(out_coord[out_ptr])
                    out_ptr += 1
            out_td[out_flat_ind] = _lookup_dep(data_td, data_shape, tuple(in_coord))
        return {out_names[0]: out_td}

    if op_type == "MatMul":
        if len(in_names) != 2:
            return _fallback_union_handler(model, node, tensor_deps, include_constants)
        a_name, b_name = in_names
        a_shape = in_shapes[a_name]
        b_shape = in_shapes[b_name]
        out_shape = out_shapes[out_names[0]]
        if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
            return _fallback_union_handler(model, node, tensor_deps, include_constants)
        m, k_a = a_shape
        k_b, n = b_shape
        if k_a != k_b:
            return _fallback_union_handler(model, node, tensor_deps, include_constants)
        k = k_a

        a_td = tensor_deps.get(a_name, _empty_tensor_deps(a_shape))
        b_td = tensor_deps.get(b_name, _empty_tensor_deps(b_shape))
        a_init = model.get_initializer(a_name)
        b_init = model.get_initializer(b_name)
        out_td = _empty_tensor_deps(out_shape)
        for out_flat_ind in range(_numel(out_shape)):
            i, j = tuple(np.unravel_index(out_flat_ind, out_shape))
            deps = set()
            for kk in range(k):
                a_zero = (a_init is not None) and (float(np.asarray(a_init)[i, kk]) == 0.0)
                b_zero = (b_init is not None) and (float(np.asarray(b_init)[kk, j]) == 0.0)

                # A(i,kk) contributes to output only when B(kk,j) is not a known zero.
                if (not b_zero) and (a_name in in_dyn):
                    deps |= _lookup_dep(a_td, a_shape, (i, kk))
                # B(kk,j) contributes to output only when A(i,kk) is not a known zero.
                if (not a_zero) and (b_name in in_dyn):
                    deps |= _lookup_dep(b_td, b_shape, (kk, j))
            out_td[out_flat_ind] = deps
        return {out_names[0]: out_td}

    return _fallback_union_handler(model, node, tensor_deps, include_constants)


def _propagate_local_tensor_deps(local_td: TensorDeps, tensor_deps: dict[str, TensorDeps]) -> TensorDeps:
    out_td: TensorDeps = []
    for local_elem_deps in local_td:
        elem_deps: ElemDeps = set()
        for src_tname, src_ind in local_elem_deps:
            src_td = tensor_deps.get(src_tname, [])
            if 0 <= src_ind < len(src_td):
                elem_deps |= src_td[src_ind]
        out_td.append(elem_deps)
    return out_td


def symbolic_dependencies(
    model,
    apply_to_subgraphs=False,
    include_constants=False,
    stop_at_quantizers=True,
    prune_zero_weights=True,
):
    """Compute fine-grained symbolic dependencies for tensor elements.

    Each tensor element is mapped to the set of source tensor elements that can
    influence it, represented as ``(tensor_name, flat_element_index)`` tuples.
    The returned analysis result contains both global tensor-level dependency
    maps and per-node local dependency maps.

    Args:
        model: The QONNX ``ModelWrapper`` to analyze. All tensor shapes in the
            model must already be resolved.
        apply_to_subgraphs: Reserved for future use. Currently ignored.
        include_constants: If ``True``, initializer tensors are treated as
            dependency sources. If ``False``, constant tensors do not contribute
            source elements.
        stop_at_quantizers: If ``True``, quantizer outputs become new symbolic
            sources and dependencies are not propagated through ``Quant`` and
            ``BipolarQuant`` nodes.
        prune_zero_weights: If ``True``, known zero-valued constant weights in
            ``Mul`` and ``MatMul`` do not contribute dependencies.

    Returns:
        A dictionary with a single ``"symbolic_dependencies"`` entry containing
        tensor-level dependencies, per-node local dependencies, and the analysis
        configuration used for the run.

    Raises:
        AssertionError: If any tensor in the model does not have a resolved
            shape.
    """

    _ = apply_to_subgraphs
    local_src_deps: dict[str, TensorDeps] = {}
    tensor_deps: dict[str, TensorDeps] = {}
    node_local: dict[str, dict[str, TensorDeps]] = {}

    all_tensors = set(model.get_all_tensor_names())

    for tname in all_tensors:
        tshape = model.get_tensor_shape(tname)
        assert tshape is not None, f"symbolic_dependencies requires resolved shape for tensor '{tname}'"
        if _is_constant_input(model, tname) and (not include_constants):
            local_src_deps[tname] = _empty_tensor_deps(tshape)
            tensor_deps[tname] = _empty_tensor_deps(tshape)
        else:
            local_src_deps[tname] = _identity_tensor_deps(tname, tshape)

        if model.find_producer(tname) is None:
            tensor_deps[tname] = _identity_tensor_deps(tname, tshape)

    # Pass 1: compute only node-local dependencies (no backward propagation).
    for node_ind, node in enumerate(model.graph.node):
        node_key = node.name if node.name != "" else f"{node_ind}:{node.op_type}"
        node_local[node_key] = _local_symbolic_handler(
            model,
            node,
            local_src_deps,
            include_constants=include_constants,
            prune_zero_weights=prune_zero_weights,
        )

    # Pass 2: propagate dependencies through the graph to get global deps.
    for node_ind, node in enumerate(model.graph.node):
        node_key = node.name if node.name != "" else f"{node_ind}:{node.op_type}"
        if stop_at_quantizers and (node.op_type in _QUANTIZER_OPS):
            for o_name in node.output:
                o_shape = _get_input_shape(model, o_name)
                tensor_deps[o_name] = _identity_tensor_deps(o_name, o_shape)
        else:
            for o_name, local_o_deps in node_local[node_key].items():
                tensor_deps[o_name] = _propagate_local_tensor_deps(local_o_deps, tensor_deps)

    return {
        "symbolic_dependencies": {
            "tensor_dependencies": tensor_deps,
            "node_local_dependencies": node_local,
            "config": {
                "include_constants": include_constants,
                "stop_at_quantizers": stop_at_quantizers,
                "prune_zero_weights": prune_zero_weights,
            },
        }
    }