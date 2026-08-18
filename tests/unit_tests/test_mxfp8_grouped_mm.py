# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import pytest
import torch
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")

import torchtitan.components.quantization.mxfp8_grouped_mm as grouped_mm  # noqa: E402
from torchtitan.components.quantization.mx import (  # noqa: E402
    _get_mxfp8_grouped_experts_cls,
)
from torchtitan.models.common.moe import GroupedExperts  # noqa: E402
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts  # noqa: E402


def _fake_quantize_activation(
    calls: list,
    saved_cols: list,
):
    def quantize(input_RK, offsets_E, *, rowwise=True, colwise=True):
        del offsets_E
        calls.append((input_RK, rowwise, colwise))
        row_RK = input_RK.clone() if rowwise else input_RK.new_empty(0)
        col_RK = input_RK.t().contiguous().t() if colwise else input_RK.new_empty(0)
        if colwise:
            saved_cols.append(col_RK)
        return grouped_mm._QuantizedActivation(
            row_RK,
            col_RK,
            input_RK.new_ones(1),
            input_RK.new_ones(1),
        )

    return quantize


def _fake_quantize_weight_rowwise(weight_t_EKN):
    return weight_t_EKN.clone(), weight_t_EKN.new_ones(1)


def _fake_quantize_weight_colwise(weight_t_EKN):
    return weight_t_EKN.transpose(-2, -1).clone(), weight_t_EKN.new_ones(1)


def _fake_scaled_grouped_mm(A, B_t, A_scales, B_scales, offsets_E):
    del A_scales, B_scales
    return torch._grouped_mm(A, B_t, offs=offsets_E, out_dtype=torch.bfloat16)


def _install_fake_kernels(monkeypatch):
    quantize_calls = []
    saved_cols = []
    monkeypatch.setattr(
        grouped_mm,
        "_quantize_activation",
        _fake_quantize_activation(quantize_calls, saved_cols),
    )
    monkeypatch.setattr(
        grouped_mm, "_quantize_weight_rowwise", _fake_quantize_weight_rowwise
    )
    monkeypatch.setattr(
        grouped_mm, "_quantize_weight_colwise", _fake_quantize_weight_colwise
    )
    monkeypatch.setattr(grouped_mm, "_scaled_grouped_mm", _fake_scaled_grouped_mm)
    return quantize_calls, saved_cols


def test_swiglu_experts_share_quantized_input_and_do_not_save_bf16_input(
    monkeypatch,
):
    quantize_calls, saved_cols = _install_fake_kernels(monkeypatch)

    MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)
    reference = GroupedExperts.Config(
        dim=32,
        hidden_dim=32,
        num_experts=2,
    ).build()
    actual = MXFP8GroupedExperts.Config(  # type: ignore[attr-defined]
        dim=32,
        hidden_dim=32,
        num_experts=2,
    ).build()

    torch.manual_seed(0)
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.1)
    actual.load_state_dict(reference.state_dict())

    reference_input = torch.randn(64, 32, dtype=torch.bfloat16, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_()
    num_tokens_per_expert_E = torch.tensor([32, 32])

    reference_output = reference(reference_input, num_tokens_per_expert_E)
    saved_ptrs = []

    def pack_hook(tensor):
        saved_ptrs.append(tensor.data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        actual_output = actual(actual_input, num_tokens_per_expert_E)

    assert len(quantize_calls) == 2
    shared_input_col = saved_cols[0]
    assert saved_ptrs.count(shared_input_col.data_ptr()) == 2
    assert actual_input.data_ptr() not in saved_ptrs
    torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=0)

    grad_output = torch.randn_like(reference_output)
    reference_output.backward(grad_output)
    actual_output.backward(grad_output)

    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)
    for actual_parameter, reference_parameter in zip(
        actual.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
            rtol=1e-2,
            atol=2e-2,
        )


def test_gpt_oss_grouped_experts_uses_normal_tensor_autograd(monkeypatch):
    _install_fake_kernels(monkeypatch)
    MXFP8GptOssGroupedExperts = _get_mxfp8_grouped_experts_cls(GptOssGroupedExperts)
    reference = GptOssGroupedExperts.Config(
        dim=32,
        hidden_dim=32,
        num_experts=2,
    ).build()
    actual = MXFP8GptOssGroupedExperts.Config(  # type: ignore[attr-defined]
        dim=32,
        hidden_dim=32,
        num_experts=2,
    ).build()

    torch.manual_seed(1)
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.1)
    actual.load_state_dict(reference.state_dict())

    reference_input = torch.randn(64, 32, dtype=torch.bfloat16, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_()
    num_tokens_per_expert_E = torch.tensor([32, 32])
    reference_output = reference(reference_input, num_tokens_per_expert_E)
    actual_output = actual(actual_input, num_tokens_per_expert_E)
    torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=0)

    grad_output = torch.randn_like(reference_output)
    reference_output.backward(grad_output)
    actual_output.backward(grad_output)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)
    for actual_parameter, reference_parameter in zip(
        actual.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
            rtol=1e-2,
            atol=2e-2,
        )


def test_swiglu_grouped_experts_compile_and_nonreentrant_checkpoint(monkeypatch):
    _install_fake_kernels(monkeypatch)
    MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)

    for use_checkpoint in (False, True):
        experts = MXFP8GroupedExperts.Config(  # type: ignore[attr-defined]
            dim=32,
            hidden_dim=32,
            num_experts=2,
        ).build()
        input_RD = torch.randn(64, 32, dtype=torch.bfloat16, requires_grad=True)
        num_tokens_per_expert_E = torch.tensor([32, 32])
        if use_checkpoint:
            output_RD = checkpoint(
                experts,
                input_RD,
                num_tokens_per_expert_E,
                use_reentrant=False,
            )
        else:
            compiled_experts = cast(
                Any,
                torch.compile(
                    experts,
                    backend="eager",
                    fullgraph=True,
                ),
            )
            output_RD = compiled_experts(input_RD, num_tokens_per_expert_E)

        output_RD.sum().backward()
        assert input_RD.grad is not None
        assert all(parameter.grad is not None for parameter in experts.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
    reason="MXFP8 requires SM100 or later",
)
def test_fused_quantization_matches_torchao_grouped_scale_contracts():
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        mx_block_rearrange_2d_M_groups_cuda,
        triton_mx_block_rearrange_2d_K_groups,
    )
    from torchao.prototype.mx_formats.config import (
        MXFP8Dim1CastKernelChoice,
        ScaleCalculationMode,
    )
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
    from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
    from torchao.quantization.quantize_.common import KernelPreference

    torch.manual_seed(0)
    input_RK = torch.randn(384, 128, device="cuda", dtype=torch.bfloat16)
    offsets_E = torch.tensor([128, 384], device="cuda", dtype=torch.int32)

    actual = grouped_mm._quantize_activation(input_RK, offsets_E)
    row_RK, row_scales = triton_to_mxfp8_dim0(input_RK, 32, "rceil")
    row_scales = mx_block_rearrange_2d_M_groups_cuda(row_scales, offsets_E)
    col_KR = _to_mxfp8_dim1_kernel_wrapper(
        input_RK,
        32,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=input_RK.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    col_scales = triton_mx_block_rearrange_2d_K_groups(
        col_KR.scale,
        offsets_E // 32,
    )

    assert torch.equal(actual.row_RK, row_RK)
    assert torch.equal(actual.row_scales, row_scales)
    assert torch.equal(actual.col_RK.t(), col_KR.qdata)
    assert torch.equal(actual.col_scales, col_scales)
