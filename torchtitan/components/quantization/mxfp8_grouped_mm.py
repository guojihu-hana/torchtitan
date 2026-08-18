# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 grouped GEMM training with normal tensor operands.

The common SwiGLU expert path quantizes its routed activation once into both
rowwise and columnwise MXFP8 representations. Gate and up grouped GEMMs share
that quantization. Each grouped GEMM saves only the columnwise representation
needed by wgrad, while the BF16 activation is allowed to die after forward.

Shape suffix legend:
    E: experts, R: routed tokens, K: input features, N: output features.
"""

from typing import NamedTuple

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.distributed.tensor import DTensor

from torchao.prototype.moe_training.kernels.mxfp8 import (
    mx_block_rearrange_2d_M_groups_cuda,
    mxfp8_quantize_cuda_3d,
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_to_mxfp8_dim0,
)

from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend


_MXFP8_BLOCK_SIZE = 32
_MXFP8_SCALING_MODE = "rceil"


class _QuantizedActivation(NamedTuple):
    row_RK: torch.Tensor  # noqa: N815
    col_RK: torch.Tensor  # noqa: N815
    row_scales: torch.Tensor
    col_scales: torch.Tensor


def _validate_grouped_mm_inputs(
    input_RK: torch.Tensor,
    weight_t_EKN: torch.Tensor,
    offsets_E: torch.Tensor,
) -> None:
    if input_RK.dtype != torch.bfloat16 or weight_t_EKN.dtype != torch.bfloat16:
        raise ValueError(
            "MXFP8 grouped GEMM requires BF16 activations and weights; "
            f"got {input_RK.dtype} and {weight_t_EKN.dtype}."
        )
    if input_RK.ndim != 2 or weight_t_EKN.ndim != 3:
        raise ValueError(
            "MXFP8 grouped GEMM requires a 2D activation and 3D expert weight; "
            f"got {input_RK.ndim}D and {weight_t_EKN.ndim}D."
        )
    if input_RK.shape[1] != weight_t_EKN.shape[1]:
        raise ValueError(
            "MXFP8 grouped GEMM contraction dimensions must match; got "
            f"{input_RK.shape[1]} and {weight_t_EKN.shape[1]}."
        )
    if offsets_E.ndim != 1 or offsets_E.dtype != torch.int32:
        raise ValueError(
            "MXFP8 grouped GEMM offsets must be a 1D int32 tensor; got "
            f"shape {tuple(offsets_E.shape)} and dtype {offsets_E.dtype}."
        )
    if offsets_E.numel() != weight_t_EKN.shape[0]:
        raise ValueError(
            "MXFP8 grouped GEMM requires one offset per expert; got "
            f"{offsets_E.numel()} offsets and {weight_t_EKN.shape[0]} experts."
        )
    for name, value in (
        ("routed token rows", input_RK.shape[0]),
        ("input features", weight_t_EKN.shape[1]),
        ("output features", weight_t_EKN.shape[2]),
    ):
        if value % _MXFP8_BLOCK_SIZE:
            raise ValueError(
                f"MXFP8 grouped GEMM requires {name} divisible by "
                f"{_MXFP8_BLOCK_SIZE}; got {value}."
            )


def _quantize_activation(
    input_RK: torch.Tensor,
    offsets_E: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = True,
) -> _QuantizedActivation:
    row_RK, col_RK, row_scales, col_scales = mxfp8_quantize_cuda(
        input_RK,
        rowwise=rowwise,
        colwise=colwise,
        scaling_mode=_MXFP8_SCALING_MODE,
    )
    if rowwise:
        row_scales = mx_block_rearrange_2d_M_groups_cuda(row_scales, offsets_E)
    if colwise:
        col_scales = triton_mx_block_rearrange_2d_K_groups(
            col_scales, offsets_E // _MXFP8_BLOCK_SIZE
        )
    return _QuantizedActivation(row_RK, col_RK, row_scales, col_scales)


def _quantize_weight_rowwise(
    weight_t_EKN: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_row_ENK, weight_row_scales = triton_to_mxfp8_dim0(
        weight_t_EKN.transpose(-2, -1),
        _MXFP8_BLOCK_SIZE,
        _MXFP8_SCALING_MODE,
    )
    return (
        weight_row_ENK.transpose(-2, -1),
        triton_mx_block_rearrange_per_group_3d(weight_row_scales),
    )


def _quantize_weight_colwise(
    weight_t_EKN: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp8_quantize_cuda_3d(
        weight_t_EKN.transpose(-2, -1),
        _MXFP8_BLOCK_SIZE,
        scale_block_dim1=_MXFP8_BLOCK_SIZE,
        scale_block_dim2=1,
        scaling_mode=_MXFP8_SCALING_MODE,
    )


def _scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    offsets_E: torch.Tensor,
) -> torch.Tensor:
    return torch._scaled_grouped_mm(
        A,
        B_t,
        A_scales,
        B_scales,
        offs=offsets_E,
        out_dtype=torch.bfloat16,
    )


@torch._dynamo.allow_in_graph
class _MXFP8GroupedMMFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        input_RK: torch.Tensor,
        weight_t_EKN: torch.Tensor,
        offsets_E: torch.Tensor,
        input_row_RK: torch.Tensor,
        input_col_RK: torch.Tensor,
        input_row_scales: torch.Tensor,
        input_col_scales: torch.Tensor,
    ) -> torch.Tensor:
        _validate_grouped_mm_inputs(input_RK, weight_t_EKN, offsets_E)

        weight_row_t_EKN, weight_row_scales = _quantize_weight_rowwise(weight_t_EKN)
        output_RN = _scaled_grouped_mm(
            input_row_RK,
            weight_row_t_EKN,
            input_row_scales,
            weight_row_scales,
            offsets_E,
        )

        requires_dgrad = ctx.needs_input_grad[0]
        requires_wgrad = ctx.needs_input_grad[1]
        ctx.save_for_backward(
            input_col_RK if requires_wgrad else input_col_RK.new_empty(0),
            input_col_scales if requires_wgrad else input_col_scales.new_empty(0),
            weight_t_EKN if requires_dgrad else weight_t_EKN.new_empty(0),
            offsets_E,
        )
        ctx.requires_dgrad = requires_dgrad
        ctx.requires_wgrad = requires_wgrad
        return output_RN

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output_RN: torch.Tensor):
        input_col_RK, input_col_scales, weight_t_EKN, offsets_E = ctx.saved_tensors

        grad_input_RK = None
        grad_weight_t_EKN = None
        if ctx.requires_dgrad or ctx.requires_wgrad:
            grad_output = _quantize_activation(
                grad_output_RN.contiguous(),
                offsets_E,
                rowwise=ctx.requires_dgrad,
                colwise=ctx.requires_wgrad,
            )

            if ctx.requires_dgrad:
                weight_col_ENK, weight_col_scales = _quantize_weight_colwise(
                    weight_t_EKN
                )
                grad_input_RK = _scaled_grouped_mm(
                    grad_output.row_RK,
                    weight_col_ENK,
                    grad_output.row_scales,
                    weight_col_scales,
                    offsets_E,
                )

            if ctx.requires_wgrad:
                grad_weight_ENK = _scaled_grouped_mm(
                    grad_output.col_RK.t(),
                    input_col_RK,
                    grad_output.col_scales,
                    input_col_scales,
                    offsets_E,
                )
                grad_weight_t_EKN = grad_weight_ENK.transpose(-2, -1)

        return (
            grad_input_RK,
            grad_weight_t_EKN,
            None,
            None,
            None,
            None,
            None,
        )


spmd.register_local_autograd_function(_MXFP8GroupedMMFunction)


def mxfp8_grouped_mm(
    input_RK: torch.Tensor,
    weight_t_EKN: torch.Tensor,
    offsets_E: torch.Tensor,
    *,
    quantized_input: _QuantizedActivation | None = None,
) -> torch.Tensor:
    """Run an MXFP8 grouped GEMM, optionally reusing a quantized activation."""
    _validate_grouped_mm_inputs(input_RK, weight_t_EKN, offsets_E)
    if quantized_input is None:
        quantized_input = _quantize_activation(input_RK, offsets_E)
    return _MXFP8GroupedMMFunction.apply(
        input_RK,
        weight_t_EKN,
        offsets_E,
        *quantized_input,
    )


def mxfp8_swiglu_grouped_experts_forward(
    experts,
    x_RD: torch.Tensor,
    num_tokens_per_expert_E: torch.Tensor,
) -> torch.Tensor:
    """Common ``GroupedExperts`` forward with shared activation quantization."""
    if isinstance(experts.w1_EFD, DTensor):
        w1_EFD = experts.w1_EFD.to_local()
        assert isinstance(experts.w2_EDF, DTensor)
        w2_EDF = experts.w2_EDF.to_local()
        assert isinstance(experts.w3_EFD, DTensor)
        w3_EFD = experts.w3_EFD.to_local()
    else:
        w1_EFD = experts.w1_EFD
        w2_EDF = experts.w2_EDF
        w3_EFD = experts.w3_EFD

    offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
    if (
        get_spmd_backend() == "spmd_types"
        and spmd.is_type_checking()
        and spmd_mesh_size("ep") == 1
    ):
        for axis in ("dp", "cp"):
            spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

    input_RD = x_RD.bfloat16()
    quantized_input = _quantize_activation(input_RD, offsets_E)
    gate_RF = mxfp8_grouped_mm(
        input_RD,
        w1_EFD.bfloat16().transpose(-2, -1),
        offsets_E,
        quantized_input=quantized_input,
    )
    up_RF = mxfp8_grouped_mm(
        input_RD,
        w3_EFD.bfloat16().transpose(-2, -1),
        offsets_E,
        quantized_input=quantized_input,
    )
    h_RF = F.silu(gate_RF) * up_RF
    output_RD = mxfp8_grouped_mm(
        h_RF,
        w2_EDF.bfloat16().transpose(-2, -1),
        offsets_E,
    )
    return output_RD.type_as(x_RD)


__all__ = ["mxfp8_grouped_mm", "mxfp8_swiglu_grouped_experts_forward"]
