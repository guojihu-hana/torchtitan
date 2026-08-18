# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels.mxfp8")

from torchao.prototype.moe_training.kernels.mxfp8 import (  # noqa: E402
    mxfp8_quantize_cuda_3d,
)
from torchtitan.components.quantization.mxfp8_fsdp import (  # noqa: E402
    _quantize_weight_32x32,
    MXFP8FSDPComputeWeight,
    MXFP8FSDPWeight,
)
from torchtitan.components.quantization.mxfp8_linear import MXFP8Linear  # noqa: E402


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="MXFP8 requires SM100 or later",
    ),
]


def _make_mxfp8_linear(
    in_features: int = 128,
    out_features: int = 96,
    *,
    bias: bool = True,
) -> MXFP8Linear:
    return (
        MXFP8Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        .build()
        .cuda()
        .bfloat16()
    )


def test_mxfp8_linear_saves_only_quantized_tensors():
    linear = _make_mxfp8_linear()
    x = torch.randn(
        37,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        output = linear(x)
        output.backward(torch.randn_like(output))

    assert output.shape == (37, linear.out_features)
    assert len(saved_tensors) == 4
    assert sum(tensor.dtype == torch.float8_e4m3fn for tensor in saved_tensors) == 2
    assert sum(tensor.dtype == torch.float8_e8m0fnu for tensor in saved_tensors) == 2
    assert all(type(tensor) is torch.Tensor for tensor in saved_tensors)
    assert isinstance(linear.weight, MXFP8FSDPWeight)


def test_mxfp8_weight_dgrad_qdata_is_transpose_view():
    weight_NK = torch.randn(
        96,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    (
        q_weight_fprop_KN,
        s_weight_fprop_blocked,
        s_weight_dgrad_blocked,
    ) = _quantize_weight_32x32(weight_NK)
    compute_weight = MXFP8FSDPComputeWeight(
        q_weight_fprop_KN,
        s_weight_fprop_blocked,
        s_weight_dgrad_blocked,
        logical_shape=weight_NK.shape,
        logical_stride=weight_NK.stride(),
        logical_storage_offset=int(weight_NK.storage_offset()),
        orig_dtype=weight_NK.dtype,
    )

    q_weight_dgrad_NK = compute_weight.q_weight_dgrad_NK
    q_weight_dgrad_reference_1NK, _ = mxfp8_quantize_cuda_3d(
        weight_NK.unsqueeze(0),
        scale_block_dim1=32,
        scale_block_dim2=32,
        scaling_mode="rceil",
        blocked_scale_output=True,
    )

    assert len(compute_weight.cache_tensors()) == 3
    assert q_weight_dgrad_NK.data_ptr() == q_weight_fprop_KN.data_ptr()
    assert torch.equal(q_weight_dgrad_NK, q_weight_dgrad_reference_1NK[0])


def test_mxfp8_linear_rejects_unwrapped_weight():
    linear = _make_mxfp8_linear()
    linear.weight = nn.Parameter(linear.weight._data.detach().clone())
    x = torch.randn(
        32,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with pytest.raises(TypeError, match="expected an MXFP8FSDPComputeWeight"):
        linear(x)


def test_mxfp8_linear_compiles_forward_and_backward():
    linear = _make_mxfp8_linear(bias=False)
    compiled_linear = torch.compile(linear, fullgraph=True)
    x = torch.randn(
        64,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output = compiled_linear(x)
    output.backward(torch.randn_like(output))

    assert output.shape == (64, linear.out_features)
    assert x.grad is not None
    assert linear.weight.grad is not None
    assert type(linear.weight.grad) is torch.Tensor


def test_mxfp8_linear_nonreentrant_checkpoint():
    linear = _make_mxfp8_linear(bias=False)
    x = torch.randn(
        64,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output = checkpoint(linear, x, use_reentrant=False)
    output.backward(torch.randn_like(output))

    assert x.grad is not None
    assert linear.weight.grad is not None
