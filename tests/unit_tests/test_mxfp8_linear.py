# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")

from torchao.prototype.moe_training.mxfp8_fsdp import MXFP8FSDPWeight  # noqa: E402
from torchao.prototype.moe_training.mxfp8_linear import (  # noqa: E402
    MXFP8Linear as TorchAOMXFP8Linear,
)

from torchtitan.components.quantization.mxfp8_linear import MXFP8Linear  # noqa: E402


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="MXFP8 requires SM100 or later",
    ),
]


def _make_torchtitan_linear(
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


def test_mxfp8_linear_saves_only_normal_quantized_tensors():
    linear = _make_torchtitan_linear()
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
    assert len(saved_tensors) == 3
    assert {tensor.dtype for tensor in saved_tensors} == {
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    }
    saved_quantized = [
        tensor for tensor in saved_tensors if tensor.dtype != torch.bfloat16
    ]
    assert all(type(tensor) is torch.Tensor for tensor in saved_quantized)
    saved_bf16 = [tensor for tensor in saved_tensors if tensor.dtype == torch.bfloat16]
    assert len(saved_bf16) == 1
    assert isinstance(linear.weight, MXFP8FSDPWeight)
    assert saved_bf16[0].data_ptr() == linear.weight._data.data_ptr()
    assert saved_bf16[0].shape != x.shape


def test_mxfp8_linear_matches_torchao_backward():
    in_features = 128
    out_features = 96
    num_tokens = 64

    torch.manual_seed(0)
    reference = (
        TorchAOMXFP8Linear(
            in_features,
            out_features,
            bias=True,
        )
        .cuda()
        .bfloat16()
    )
    actual = _make_torchtitan_linear(in_features, out_features, bias=True)
    with torch.no_grad():
        actual.weight.copy_(reference.weight)
        actual.bias.copy_(reference.bias)

    reference_input = torch.randn(
        num_tokens,
        in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    actual_input = reference_input.detach().clone().requires_grad_()

    reference_output = reference(reference_input)
    actual_output = actual(actual_input)
    torch.testing.assert_close(actual_output, reference_output, rtol=1e-2, atol=1e-2)

    grad_output = torch.randn_like(reference_output)
    reference_output.backward(grad_output)
    actual_output.backward(grad_output)

    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.weight.grad, reference.weight.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad, rtol=0, atol=0)


def test_mxfp8_linear_compiles_forward_and_backward():
    linear = _make_torchtitan_linear(bias=False)
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


def test_mxfp8_linear_nonreentrant_checkpoint():
    linear = _make_torchtitan_linear(bias=False)
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
