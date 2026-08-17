### <a name="LookupTable"></a><a name="abs">**LookupTable**</a>

Evaluates a set of lookup tables (LUTs) over the last axis of an input tensor, and produces one output data. Each of
the `M` neurons in the layer draws a fixed-size fan-in of `K` elements from the last axis of the input, packs those
values into an unsigned integer address, and emits the table entry found at that address.


#### Version

This operator is not part of the ONNX standard.
The description of this operator in this document corresponds to `qonnx.custom_op.lnn` opset version 3.

#### Attributes

<dl>
<dt><tt>input_bits</tt> : int (default is 1)</dt>
<dd>The number of bits per input element; each element of X must be an integer value in [0, 2**input_bits).</dd>
<dt><tt>out_bits</tt> : int (default is 0)</dt>
<dd>The number of output bits Y, when narrower than the full bit-width of table's element type. The default, 0, means Y equals the full bit-width of table's element type.</dd>
</dl>

#### Inputs

<dl>
<dt><tt>X</tt> : tensor(bool), tensor(uint8), tensor(uint16), tensor(uint32)</dt>
<dd>Input tensor of shape [..., C_in], containing unsigned integer values in [0, 2**input_bits). The lookup is performed over the last axis; all leading axes are treated as batch axes.</dd>
<dt><tt>indices</tt> : tensor(int64), tensor(int32)</dt>
<dd>Connectivity tensor of shape [M, K], where M is the number of neurons and K the fan-in of every neuron in the node. Element [m, k] is the position along the last axis of X that feeds slot k of neuron m, and must be in [0, C_in). The position k determines the significance of the slot in the address, so the order of entries is meaningful.</dd>
<dt><tt>table</tt> : tensor(bool), tensor(uint8), tensor(int8), tensor(uint16), tensor(int16), tensor(uint32), tensor(int32)</dt>
<dd>Truth table tensor of shape [M, S], where S = 2**(K * input_bits). Element [m, s] is the (possibly multi-bit) output of neuron m when its address evaluates to s, stored as a plain integer value of the tensor's element type; this same type is used for the operator's output.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>Y</tt></dt>
<dd>Output tensor of shape [..., M], with the same element type as table and leading axes matching those of X. Element [..., m] is the output of neuron m.</dd>
</dl>

#### Address computation

The address of neuron `m` is formed by concatenating the bits of its fan-in values, with slot `k` occupying bits
`[k * input_bits, (k+1) * input_bits)`:

```
addr[..., m] = sum over k of  X[..., indices[m, k]] * 2**(input_bits * k)
Y[..., m] = table[m, addr[..., m]]
```

For a two-input, one-bit neuron this yields the conventional truth table ordering, where `table[m]` is indexed by
`a0 + 2*a1`:

| a1 | a0 | address | AND | OR | XOR |
|----|----|---------|-----|----|-----|
| 0  | 0  | 0       | 0   | 0  | 0   |
| 0  | 1  | 1       | 0   | 1  | 1   |
| 1  | 0  | 2       | 0   | 1  | 1   |
| 1  | 1  | 3       | 1   | 1  | 0   |

#### Notes

This is the core operator for representing logic gate networks and LUT-based neural networks (LNNs), such as
LogicNets, DiffLogicNet, PolyLUT and NeuraLUT, as well as their hybrids with regular quantized neural networks. A
single operator covers logic gate networks (`K=2`, `input_bits=1`), sparse LUT layers, and — in combination with
`Im2Col` — convolutional LNNs.

* The operator implements X:Y LUTs, where the input width is `X = K * input_bits`, and the output width `Y` is given
  by `out_bits` if set, or otherwise defaults to the full bit-width of the table's element type (e.g. up to 8 bits
  for a `uint8` table).
* Every neuron in a `LookupTable` node has the same fan-in `K`. A layer with neurons of different fan-in is
  represented as several `LookupTable` nodes, one per distinct fan-in, whose outputs are concatenated.
* `tensor(bool)` for `X` is only meaningful with `input_bits=1`, since a bool element cannot carry more than one bit.
* `out_bits` is purely declarative: it does not affect the computed output, only documents and allows validating the
  value range of `table` (all entries must be in [0, 2**out_bits) when set), which downstream consumers such as
  hardware backends rely on to size the actual output signal.
* Pick the narrowest `table` type that can hold the neuron's output range, e.g. `tensor(bool)` or `tensor(uint8)` for
  a 1-bit output, `tensor(uint8)` for up to 8 output bits.
* Floating-point tensors are not accepted for `X`. Chaining the output of one `LookupTable` node into another needs
  no conversion when the upstream `table`'s dtype already matches one of `X`'s allowed types. To enter this integer
  domain from a real-valued, QONNX-quantized activation, convert `Quant`'s float32 output with an explicit `Cast`
  (or use `QuantizeLinear`/`DequantizeLinear` instead of `Quant` if an affine 8-bit quantization scheme suffices).
* A convolutional LNN layer is expressed as `Im2Col` followed by `LookupTable`, with `indices` then selecting
  positions from within the resulting patch.
* Scaled activations (e.g. `{0, 0.5}`) are expressed by storing integer levels in `table` and following the node
  with `Cast` to float and `Mul` (and `Add` for a zero-point), in line with the integer-plus-scale convention used
  by the QONNX quantization operators.
* The operator is not differentiable with respect to `table`; it is intended for representing an already-trained
  network, not for training in ONNX.

#### Examples

<details>
<summary>LookupTable</summary>

```python
from onnx import helper
import numpy as np

# a layer of three two-input gates over four binary inputs: XOR, AND, OR
indices = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
table = np.array([
    [0, 1, 1, 0],   # XOR
    [0, 0, 0, 1],   # AND
    [0, 1, 1, 1],   # OR
], dtype=np.uint8)
x = np.array([[0, 1, 1, 0]], dtype=np.uint8)

# Create node
node = helper.make_node(
    'LookupTable',
    domain='qonnx.custom_op.lnn',
    inputs=['x', 'indices', 'table'],
    outputs=['y'],
    input_bits=1,
)

# Execute the same settings with the reference implementation
output_ref = lookup_table(x, indices, table, input_bits=1)
# -> [[1, 1, 1]] : XOR(0,1)=1, AND(1,1)=1, OR(1,0)=1

# Execute node and compare
expect(node, inputs=[x, indices, table], outputs=[output_ref], name='test_lookuptable')
```

A 6:2 LUT, whose 2-bit output is stored directly as a `uint8` table entry with no separate output axis, followed by
the `Cast`+`Mul` needed to turn it into a scaled float32 activation (e.g. levels 0, 0.5, 1.0, 1.5). `out_bits=2`
declares that only the lower 2 bits of the `uint8` container are actually used:

```python
# K=6, input_bits=1 -> S=64 addresses; each entry is a 2-bit value in {0,1,2,3}
table = np.random.randint(0, 4, size=(M, 64)).astype(np.uint8)
lut = helper.make_node('LookupTable', domain='qonnx.custom_op.lnn',
                       inputs=['x', 'indices', 'table'], outputs=['y_bits'], input_bits=1, out_bits=2)
cast = helper.make_node('Cast', inputs=['y_bits'], outputs=['y_f'], to=TensorProto.FLOAT)
scale = helper.make_node('Mul', inputs=['y_f', 'out_scale'], outputs=['y'])
```

A layer with neurons of mixed fan-in is represented as one `LookupTable` node per distinct fan-in value, with the
node outputs concatenated in the desired neuron order:

```python
# 2 neurons with fan-in 3, 1 neuron with fan-in 2, over the same input
node_k3 = helper.make_node('LookupTable', domain='qonnx.custom_op.lnn',
                           inputs=['x', 'indices_k3', 'table_k3'], outputs=['y_k3'])
node_k2 = helper.make_node('LookupTable', domain='qonnx.custom_op.lnn',
                           inputs=['x', 'indices_k2', 'table_k2'], outputs=['y_k2'])
concat = helper.make_node('Concat', inputs=['y_k3', 'y_k2'], outputs=['y'], axis=-1)
```

</details>

#### Sample Implementation

<details>
<summary>LookupTable</summary>

```python
# SPDX-License-Identifier: Apache-2.0

import numpy as np


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
    """table has shape [M, S]; Y has table's dtype and shape [..., M]."""
    M, K = indices.shape
    S = 2 ** (K * input_bits)
    dense = table.reshape(M, S)
    addr = lut_address(x, indices, input_bits)  # [..., M]
    flat = addr + np.arange(M, dtype=np.int64) * S
    return dense.reshape(M * S)[flat]  # [..., M], dtype preserved
```

</details>

#### Decomposition into standard ONNX operators

<details>
<summary>LookupTable as an ONNX function</summary>

The operator can be expressed as an ONNX `FunctionProto` built only from standard ONNX operators, so that a model may
keep the custom node structure while remaining executable by runtimes that do not know the `qonnx.custom_op.lnn`
domain.

```
ib          = Constant(value_int = <ref to input_bits attribute>)
M, K        = Gather(Shape(indices), [0]), Gather(Shape(indices), [1])
S           = Gather(Shape(table), [1])

# per-slot address weights 2**(input_bits*k), computed as plain int64 (Pow
# accepts integer base/exponent, so no float round-trip is needed)
wt          = Pow(2, Mul(Range(0, K, 1), ib))

# gather the fan-in values (X's dtype) and reduce them to an int64 address
fanin       = Cast(Gather(X, indices, axis=-1), int64)        # [..., M, K]
addr        = ReduceSum(Mul(fanin, wt), [-1], keepdims=0)     # int64

# flatten (neuron, address) into a single gather over the fully-flattened table;
# the result already has shape [..., M], matching indices' shape, so no reshape is needed
flat_addr   = Add(addr, Mul(Range(0, M, 1), S))
tbl1d       = Reshape(table, [-1])                           # [M*S]
Y           = Gather(tbl1d, flat_addr, axis=0)               # [..., M], table's dtype
```

Since `Gather` preserves the element type of `table`, `Y` naturally has `table`'s dtype and no final `Cast` is
needed. Note that `uint4`/`int4` tables, although expressible in ONNX, are rejected by current onnxruntime versions
and are therefore not part of the type constraints above.

</details>
