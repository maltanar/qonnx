### <a name="LookupTableConv"></a><a name="abs">**LookupTableConv**</a>

Evaluates a weight-shared tree of lookup tables (LUTs) over sliding receptive fields of a spatial input tensor,
fusing patch extraction (as done by `Im2Col`), the receptive-field connectivity, and every level of a fixed-depth
LUT tree into a single node. Each of the `num_kernels` output neurons is the root of a binary(-or-`lut_rank`-ary)
tree of depth `tree_depth`: its `lut_rank**tree_depth` leaves draw from the receptive field, and every non-leaf node
combines `lut_rank` child outputs via its own small LUT. The whole tree is evaluated identically at every
sliding-window position, i.e. weight-tied across the output spatial grid, just like a regular convolution shares its
kernel weights.

An optional passthrough mode (`tree_depth = 0`) turns the node into a pure connectivity-gather: no lookup table is
evaluated, and the node emits the raw gathered fan-in values instead.

#### Version

This operator is not part of the ONNX standard.
The description of this operator in this document corresponds to `qonnx.custom_op.lnn` opset version 1.

#### Attributes

<dl>
<dt><tt>tree_depth</tt> : int (default is 1, must be >= 0)</dt>
<dd>Number of levels in the tree. Level 0 (the leaves) draws directly from the receptive field; level <tt>tree_depth - 1</tt> is the root and produces the node's output. <tt>tree_depth = 1</tt> means a single level of leaves with no internal combination. <tt>tree_depth = 0</tt> is a passthrough mode: no lookup table is evaluated, and the node emits the raw gathered fan-in values instead (see <a href="#outputs">Outputs</a>). <tt>num_kernels</tt> (M) and <tt>lut_rank</tt> are not separate attributes — they are inferred from <tt>indices</tt>'s shape (see <a href="#inputs">Inputs</a>).</dd>
<dt><tt>kernel_shape</tt> : list of ints (required, every entry >= 1)</dt>
<dd>Spatial size of the receptive field, one entry per spatial dimension (length 2 for 2D, 3 for 3D).</dd>
<dt><tt>strides</tt> : list of ints (default is 1 for every spatial dimension, every entry >= 1)</dt>
<dd>Stride of the sliding receptive field along each spatial dimension.</dd>
<dt><tt>pads</tt> : list of ints (default is 0 for every entry, every entry >= 0)</dt>
<dd>Zero-padding applied to the input before the receptive field slides over it, following the ONNX <tt>Conv</tt> convention: <tt>[begin_1, ..., begin_n, end_1, ..., end_n]</tt>, one begin/end pair per spatial dimension.</dd>
<dt><tt>channel_group_size</tt> : int (default is 0)</dt>
<dd>Purely documentary: when non-zero, records that <tt>indices</tt> were originally sampled from contiguous groups of this many input channels (as opposed to freely across all channels). Does not affect the computed output.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : tensor(bool), tensor(uint8), tensor(uint16), tensor(uint32)</dt>
<dd>Input tensor of shape [N, D1, ..., Dn, C] (channels-last, n = 2 or 3 spatial dimensions). When <tt>tree_depth >= 1</tt>, every element must be in {0, 1} — each leaf of the tree consumes exactly one bit. When <tt>tree_depth == 0</tt> (passthrough), elements may be arbitrary integer values within the chosen type's range, since no address is ever computed from them.</dd>
<dt><tt>indices</tt> : tensor(int64), tensor(int32)</dt>
<dd>Connectivity tensor of shape [M, P, lut_rank], where <tt>M</tt> (num_kernels) is <tt>indices.shape[0]</tt> and <tt>lut_rank</tt> is <tt>indices.shape[-1]</tt>. When <tt>tree_depth >= 1</tt>, <tt>P</tt> must equal <tt>lut_rank**(tree_depth - 1)</tt>, the number of tree leaves per kernel; when <tt>tree_depth == 0</tt>, <tt>P</tt> must be 1. Element [m, p, k] addresses a position within the flattened receptive-field patch, i.e. a value in [0, C * prod(kernel_shape)), using the same row-major-spatial, channel-minor flattening order as `Im2Col`'s output columns. This same fan-in pattern is reused, unshifted, at every output position. Only the leaves need explicit connectivity: every level above them combines its children using a fixed, position-independent pairing (see <a href="#tree-evaluation">Tree evaluation</a>), so no indices are needed for internal levels.</dd>
<dt><tt>table</tt> (optional) : tensor(bool), tensor(uint8)</dt>
<dd>Tensor of shape [M, N_nodes, 2**lut_rank], where <tt>N_nodes = sum over i in [0, tree_depth) of lut_rank**i</tt> is the number of nodes in one kernel's tree (leaves and internal nodes combined). This is 0 when <tt>tree_depth == 0</tt>, so <tt>table</tt> must be omitted in that case — there are no nodes to store. Element [m, j, s] is the output of node <tt>j</tt> of kernel <tt>m</tt>'s tree when its address evaluates to <tt>s</tt>, restricted to {0, 1} since every node is a plain boolean function of its <tt>lut_rank</tt> one-bit inputs. Nodes are packed level-major starting from the leaves: the first <tt>lut_rank**(tree_depth-1)</tt> rows (in the same <tt>p</tt> order as <tt>indices</tt>' middle axis) are the leaves, the next <tt>lut_rank**(tree_depth-2)</tt> rows are level 1, and so on down to the single root row last. Every level's table has the same width <tt>2**lut_rank</tt>. <tt>table.shape[0]</tt> must equal <tt>indices.shape[0]</tt> (M).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt></dt>
<dd>
When <tt>tree_depth >= 1</tt>: tensor of shape [N, O1, ..., On, M], with the same element type as <tt>table</tt>. O1..On are the output spatial dimensions, computed as for a standard convolution: <tt>O_d = floor((D_d + pads[d] + pads[d+n] - kernel_shape[d]) / strides[d]) + 1</tt>. Element [n, o1, ..., on, m] is the root-level output of kernel m's tree at output position (o1, ..., on).<br><br>
When <tt>tree_depth == 0</tt> (passthrough): tensor of shape [N, O1, ..., On, M, lut_rank], with the same element type as <tt>X</tt>. Element [n, o1, ..., on, m, k] is the raw value gathered from the receptive field for slot [m, 0, k] of <tt>indices</tt>, at output position (o1, ..., on) — i.e. no lookup table is evaluated, only the connectivity-driven gather is performed.
</dd>
</dl>

#### <a name="tree-evaluation"></a>Tree evaluation

Let `patch[n, o, k1, ..., kn, c]` be the (implicitly zero-padded) receptive field read from `X` at output position
`o = (o1, ..., on)`:

```
patch[n, o, k1, ..., kn, c] = X_padded[n, o1*strides[0] + k1, ..., on*strides[n-1] + kn, c]
```

where `X_padded` is `X` conceptually padded with zeros according to `pads` (no separate `Pad` node is needed), and
`flat_patch[n, o, :]` is `patch[n, o]` flattened over its last `1 + n` axes in row-major-spatial, channel-minor order
(matching `Im2Col`'s column order).

When `tree_depth == 0`, no lookup table is evaluated: `Y[n, o, m, k] = flat_patch[n, o, indices[m, 0, k]]` and
none of the levels below apply. Otherwise:

**Leaves (level 0).** For kernel `m`, leaf `p`:

```
addr0[n, o, m, p] = sum over k of  flat_patch[n, o, indices[m, p, k]] * 2**k
leaf[n, o, m, p]  = table[m, p, addr0[n, o, m, p]]                      # boolean
```

**Internal levels (1 .. tree_depth - 1).** Level `l` has `lut_rank**(tree_depth-1-l)` nodes per kernel. Its inputs
are level `l-1`'s outputs for that kernel, grouped into consecutive chunks of `lut_rank` — node `q` of level `l`
consumes children `q*lut_rank .. q*lut_rank + lut_rank - 1` of level `l-1` (no indices tensor needed: this pairing is
fixed given `lut_rank`, independent of spatial position or trained weights):

```
addr_l[n, o, m, q]  = sum over k of  level_{l-1}[n, o, m, q*lut_rank + k] * 2**k
level_l[n, o, m, q] = table[m, row_offset(l) + q, addr_l[n, o, m, q]]
```

where `row_offset(l) = sum over i in [0, l) of lut_rank**(tree_depth-1-i)` locates level `l`'s block within `table`'s
node axis. The root level (`l = tree_depth - 1`) has exactly one node per kernel, and `level_{tree_depth-1}[n, o, m, 0]`
is `Y[n, o, m]`.

#### Notes

This operator represents a full convolutional logic-gate-network / LUT-based neural network layer (e.g. convolutional
variants of LogicNets, DiffLogicNet, PolyLUT, NeuraLUT), analogous to what [`LookupTable`](lookuptable_v3.md)
represents for a single fully-connected level.

* `num_kernels` (M) and `lut_rank` are inferred from `indices`'s shape rather than declared as separate attributes,
  the same convention `LookupTable` uses for M and K, so they cannot disagree with the tensors actually provided.
* The fan-in pattern in `indices` is reused, unshifted, at every output position — only the receptive field's origin
  moves, exactly like a convolution kernel shares its weights across spatial positions.
* Packing one small table per tree node (rather than a single `2**(lut_rank**tree_depth)`-wide table per kernel)
  keeps every individual LUT at the physical width hardware backends care about (e.g. `lut_rank = 2` for a tree of
  FPGA LUT2/LUT4/LUT6 primitives), and preserves the tree-decomposable-function constraint rather than silently
  generalizing to arbitrary functions of the leaves.
* Every input bit is exactly 1 bit wide, at every level, including the leaves: `X` must contain only 0/1 values
  whenever `tree_depth >= 1`. This matches how these networks are actually trained and evaluated (all intermediate
  values are boolean); it also means every level's table has the same width `2**lut_rank`, so `table` can be one
  uniformly-shaped tensor instead of needing per-level shapes. Passthrough (`tree_depth == 0`) is exempt from this,
  since no address is ever computed from `X` there — see the next bullet.
* Passthrough (`tree_depth == 0`) exists so that future, more flexible tree structures (non-uniform fan-in,
  non-canonical wiring between levels, etc.) can still use this op's receptive-field gather, without being forced
  through the fixed `lut_rank`/canonical-pairing evaluation described above. In that mode the op is equivalent to
  `Im2Col` followed by a `Gather` on `indices`, and `X` may carry arbitrary multi-bit values (e.g. a quantized
  activation) since they are only ever gathered, never addressed into a table.
* `channel_group_size` is informational only, provided so hardware backends can see how fan-in was structured;
  downstream consumers that do not care about it can ignore it.
* Floating-point tensors are not accepted for `X`, for the same reasons as `LookupTable`.
* `table`'s type constraints are narrower than `LookupTable`'s: since every node in the tree is strictly boolean
  (there is no `out_bits`-style widening any more), only `tensor(bool)` and `tensor(uint8)` are accepted.
* The operator is not differentiable with respect to `table`; it is intended for representing an already-trained
  network.
* Validity constraints a producer must satisfy (not re-derivable from a single attribute or tensor in isolation):
  `tree_depth >= 0`; every `kernel_shape`/`strides` entry `>= 1` and every `pads` entry `>= 0`; every value in
  `X` is in {0, 1} whenever `tree_depth >= 1`; every value in `indices` lies in `[0, C * prod(kernel_shape))`;
  `indices.shape[1]` equals `lut_rank**(tree_depth - 1)` when `tree_depth >= 1`, or `1` when `tree_depth == 0`;
  and, when `table` is given, `table.shape == [indices.shape[0], N_nodes, 2**lut_rank]`.

#### Examples

<details>
<summary>LookupTableConv</summary>

```python
from onnx import helper
import numpy as np

# depth-2 tree of 2-input gates, one kernel, 2x2 receptive field, stride 2, no padding, 1 input channel
# leaves: 4 inputs -> 2 leaf gates (2 inputs each) -> 1 root gate (2 inputs) combining the leaves
tree_depth = 2  # num_kernels=1, lut_rank=2 are inferred below from indices' shape (1, 2, 2)
indices = np.array([[[0, 3], [1, 2]]], dtype=np.int64)  # shape (1, 2, 2): kernel 0's two leaves
table = np.array([[
    [0, 1, 1, 0],   # leaf 0: XOR
    [0, 0, 0, 1],   # leaf 1: AND
    [0, 1, 1, 1],   # root: OR(leaf0, leaf1)
]], dtype=np.uint8)  # shape (1, 3, 4): N_nodes = (2**2 - 1)/(2 - 1) = 3
x = np.array([[[[0], [1]], [[1], [0]]]], dtype=np.uint8)  # shape (1, 2, 2, 1): N, H, W, C

node = helper.make_node(
    'LookupTableConv',
    domain='qonnx.custom_op.lnn',
    inputs=['x', 'indices', 'table'],
    outputs=['y'],
    tree_depth=tree_depth,
    kernel_shape=[2, 2],
    strides=[2, 2],
    pads=[0, 0, 0, 0],
)
# -> y shape (1, 1, 1, 1): single output position, single kernel
```

Passthrough mode (`tree_depth=0`), exposing the raw fan-in for external tree-combination logic:

```python
indices_raw = np.array([[[0, 3]]], dtype=np.int64)  # shape (1, 1, 2): single kernel, P=1, lut_rank=2

node = helper.make_node(
    'LookupTableConv',
    domain='qonnx.custom_op.lnn',
    inputs=['x', 'indices_raw'],  # table omitted
    outputs=['y_raw'],
    tree_depth=0,
    kernel_shape=[2, 2],
    strides=[2, 2],
    pads=[0, 0, 0, 0],
)
# -> y_raw shape (1, 1, 1, 1, 2): (N, O1, O2, num_kernels, lut_rank)
```

</details>

#### Sample Implementation

<details>
<summary>LookupTableConv</summary>

```python
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def _im2col_patches(x, kernel_shape, strides, pads):
    """x: [N, *D, C] -> patches: [N, *O, prod(kernel_shape) * C], row-major-spatial/channel-minor."""
    n_spatial = len(kernel_shape)
    pad_width = [(0, 0)] + [(pads[d], pads[d + n_spatial]) for d in range(n_spatial)] + [(0, 0)]
    x_padded = np.pad(x, pad_width, mode="constant", constant_values=0)
    out_shape = [
        (x.shape[1 + d] + pads[d] + pads[d + n_spatial] - kernel_shape[d]) // strides[d] + 1
        for d in range(n_spatial)
    ]
    patches = np.empty((x.shape[0], *out_shape, np.prod(kernel_shape) * x.shape[-1]), dtype=x.dtype)
    for o in np.ndindex(*out_shape):
        starts = [o[d] * strides[d] for d in range(n_spatial)]
        slices = tuple(slice(s, s + k) for s, k in zip(starts, kernel_shape))
        patch = x_padded[(slice(None), *slices, slice(None))]  # [N, *kernel_shape, C]
        patches[(slice(None), *o)] = patch.reshape(x.shape[0], -1)
    return patches


def lookup_table_conv(x, indices, table, tree_depth, kernel_shape, strides, pads):
    """indices: [M, P, lut_rank]; table: [M, N_nodes, 2**lut_rank] or None (only when tree_depth == 0).

    M and lut_rank are inferred from indices's shape, not passed separately. P must be 1 when
    tree_depth == 0 (passthrough exposes a single lut_rank-wide gather per kernel).
    """
    lut_rank = indices.shape[-1]
    patches = _im2col_patches(x, kernel_shape, strides, pads)  # [N, *O, C*prod(kernel_shape)]
    gathered = patches[..., indices]  # [N, *O, M, P, lut_rank]

    if tree_depth == 0:
        return gathered[..., 0, :]  # drop the (always size-1) P axis -> [N, *O, M, lut_rank]

    weight = 2 ** np.arange(lut_rank)
    level = gathered  # [..., M, num_nodes_this_level, lut_rank]
    row_offset = 0
    for l in range(tree_depth):
        addr = (level * weight).sum(-1)  # [..., M, num_nodes_this_level]
        num_nodes_this_level = addr.shape[-1]
        rows = table[:, row_offset:row_offset + num_nodes_this_level, :]  # [M, num_nodes, 2**lut_rank]
        out = np.take_along_axis(
            np.broadcast_to(rows, (*addr.shape[:-2], *rows.shape)),
            addr[..., None].astype(np.int64), axis=-1,
        )[..., 0]  # [..., M, num_nodes_this_level]
        row_offset += num_nodes_this_level
        if l < tree_depth - 1:
            level = out.reshape(*out.shape[:-1], num_nodes_this_level // lut_rank, lut_rank)
        else:
            return out[..., 0]  # root: exactly one node per kernel
```

</details>
