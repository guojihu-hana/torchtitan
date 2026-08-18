# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 linear training with FSDP-managed 32x32 weight caches.

Tensor shape suffixes:
    M: flattened token rows
    N: output features
    K: input features
"""

from dataclasses import dataclass, replace

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.sharding import LocalMapConfig

from .mxfp8_fsdp import _quantize_weight_32x32, MXFP8FSDPComputeWeight, MXFP8FSDPWeight


TP = MeshAxisName.TP

_MXFP8_BLOCK_SIZE = 32
_MXFP8_SCALING_MODE = "rceil"


def _pad_rows(x_MK: torch.Tensor) -> tuple[torch.Tensor, int]:
    num_rows = x_MK.shape[0]
    num_padded_rows = (
        (num_rows + _MXFP8_BLOCK_SIZE - 1) // _MXFP8_BLOCK_SIZE
    ) * _MXFP8_BLOCK_SIZE
    if num_padded_rows == num_rows:
        return x_MK, num_rows
    return F.pad(x_MK, (0, 0, 0, num_padded_rows - num_rows)), num_rows


@torch._dynamo.allow_in_graph
class _MXFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        weight_NK: torch.Tensor,
        bias_N: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(weight_NK, MXFP8FSDPComputeWeight):
            weight_shape = weight_NK.shape
            q_weight_fprop_KN = weight_NK.q_weight_fprop_KN
            s_weight_fprop_blocked = weight_NK.s_weight_fprop_blocked
            q_weight_dgrad_NK = weight_NK.q_weight_dgrad_NK
            s_weight_dgrad_blocked = weight_NK.s_weight_dgrad_blocked
        elif isinstance(weight_NK, MXFP8FSDPWeight):
            weight_hp_NK = weight_NK._data
            weight_shape = weight_hp_NK.shape
            (
                q_weight_fprop_KN,
                s_weight_fprop_blocked,
                s_weight_dgrad_blocked,
            ) = _quantize_weight_32x32(weight_hp_NK)
            # This view shares qdata storage with the FPROP orientation.
            q_weight_dgrad_NK = q_weight_fprop_KN.t()
        else:
            raise TypeError(
                "MXFP8Linear expected an MXFP8FSDPComputeWeight from FSDP or "
                "an MXFP8FSDPWeight for non-FSDP execution, "
                f"got {type(weight_NK)}."
            )

        if x.dtype != torch.bfloat16 or weight_NK.dtype != torch.bfloat16:
            raise ValueError(
                "MXFP8Linear requires BF16 activations and weights; "
                f"got activation dtype {x.dtype} and weight dtype {weight_NK.dtype}."
            )
        if bias_N is not None and bias_N.dtype != torch.bfloat16:
            raise ValueError(
                f"MXFP8Linear requires a BF16 bias; got bias dtype {bias_N.dtype}."
            )
        if x.shape[-1] != weight_shape[1]:
            raise ValueError(
                "MXFP8Linear activation and weight contraction dimensions must "
                f"match; got {x.shape[-1]} and {weight_shape[1]}."
            )
        for name, value in (
            ("local in_features", weight_shape[1]),
            ("local out_features", weight_shape[0]),
        ):
            if value % _MXFP8_BLOCK_SIZE:
                raise ValueError(
                    f"MXFP8Linear requires {name} divisible by "
                    f"{_MXFP8_BLOCK_SIZE}; got {value}."
                )

        input_shape = x.shape
        x_MK, num_rows = _pad_rows(x.reshape(-1, input_shape[-1]).contiguous())
        requires_wgrad = ctx.needs_input_grad[1]

        x_row_MK, x_col_MK, x_row_scales, x_col_scales = mxfp8_quantize_cuda(
            x_MK,
            rowwise=True,
            colwise=requires_wgrad,
            scaling_mode=_MXFP8_SCALING_MODE,
        )
        x_row_scales = triton_mx_block_rearrange(x_row_scales)
        if requires_wgrad:
            x_col_scales = triton_mx_block_rearrange(x_col_scales)

        # The 32x32 weight scales are expanded into the hardware's 1x32
        # blocked representation, so both operands use BlockWise1x32 here.
        output_MN = F.scaled_mm(
            x_row_MK,
            q_weight_fprop_KN,
            scale_a=x_row_scales,
            scale_recipe_a=F.ScalingType.BlockWise1x32,
            scale_b=s_weight_fprop_blocked,
            scale_recipe_b=F.ScalingType.BlockWise1x32,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            bias=bias_N,
            output_dtype=torch.bfloat16,
        )

        # Stash the columnwise MXFP8 activation and scales for WGRAD instead of
        # the BF16 input. The DGRAD weight is only a transpose view of the
        # FPROP qdata, so saving it does not retain a second FP8 allocation.
        # FSDP repopulates the shared qdata storage before backward if needed.
        ctx.save_for_backward(
            x_col_MK,
            x_col_scales,
            q_weight_dgrad_NK,
            s_weight_dgrad_blocked,
        )
        ctx.input_shape = input_shape
        ctx.num_rows = num_rows
        ctx.requires_dgrad = ctx.needs_input_grad[0]
        ctx.requires_wgrad = requires_wgrad
        ctx.has_bias = bias_N is not None

        return output_MN[:num_rows].reshape(*input_shape[:-1], weight_shape[0])

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        # The activation pair is consumed by WGRAD, while the weight pair is
        # consumed by DGRAD. Neither gradient path needs the original BF16 input.
        (
            x_col_MK,
            x_col_scales,
            q_weight_dgrad_NK,
            s_weight_dgrad_blocked,
        ) = ctx.saved_tensors

        grad_output_MN = grad_output.contiguous().reshape(-1, grad_output.shape[-1])
        grad_bias_N = grad_output_MN.sum(dim=0) if ctx.has_bias else None

        grad_input = None
        grad_weight_NK = None
        if ctx.requires_dgrad or ctx.requires_wgrad:
            padded_grad_output_MN, _ = _pad_rows(grad_output_MN)
            (
                grad_output_row_MN,
                grad_output_col_MN,
                grad_output_row_scales,
                grad_output_col_scales,
            ) = mxfp8_quantize_cuda(
                padded_grad_output_MN,
                rowwise=ctx.requires_dgrad,
                colwise=ctx.requires_wgrad,
                scaling_mode=_MXFP8_SCALING_MODE,
            )

            if ctx.requires_dgrad:
                grad_output_row_scales = triton_mx_block_rearrange(
                    grad_output_row_scales
                )
                grad_input_MK = F.scaled_mm(
                    grad_output_row_MN,
                    q_weight_dgrad_NK,
                    scale_a=grad_output_row_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=s_weight_dgrad_blocked,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )
                grad_input = grad_input_MK[: ctx.num_rows].reshape(ctx.input_shape)

            if ctx.requires_wgrad:
                grad_output_col_scales = triton_mx_block_rearrange(
                    grad_output_col_scales
                )
                grad_weight_NK = F.scaled_mm(
                    grad_output_col_MN.t(),
                    x_col_MK,
                    scale_a=grad_output_col_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=x_col_scales,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )

        return grad_input, grad_weight_NK, grad_bias_N


spmd.register_local_autograd_function(_MXFP8LinearFunction)


class MXFP8Linear(Linear):
    """Linear using 1x32 activations and 32x32 cached weights."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for ``Linear.Config``."""

        def __post_init__(self) -> None:
            for name in ("in_features", "out_features"):
                value = getattr(self, name)
                if value % _MXFP8_BLOCK_SIZE:
                    raise ValueError(
                        f"MXFP8 requires {name} divisible by {_MXFP8_BLOCK_SIZE}; "
                        f"got {name}={value}."
                    )

        def build(self, **kwargs):
            instance = Linear.Config.build(self, **kwargs)
            if instance._sharding_config is not None:
                sharding_config = instance._sharding_config
                weight_tp = (
                    sharding_config.state_shardings["weight"]
                    .per_axis_spmd_types()
                    .get(TP)
                )
                rowwise = isinstance(weight_tp, spmd.Shard) and weight_tp.dim == 1
                if rowwise:
                    input_layout = dense_activation_placement(tp=spmd.S(-1))
                    input_grad_layout = dense_activation_placement(tp=spmd.S(-1))
                else:
                    input_layout = dense_activation_placement(tp=spmd.R)
                    input_grad_layout = dense_activation_placement(tp=spmd.P)
                instance._sharding_config = replace(
                    sharding_config,
                    in_src_shardings={
                        **(sharding_config.in_src_shardings or {}),
                        "input": input_layout,
                    },
                    in_dst_shardings={
                        **(sharding_config.in_dst_shardings or {}),
                        "input": input_layout,
                    },
                    local_map=LocalMapConfig(in_grad_placements=(input_grad_layout,)),
                )
            return instance

    def __init__(self, config: Config):
        super().__init__(config)
        self.weight = nn.Parameter(
            MXFP8FSDPWeight(self.weight),
            requires_grad=self.weight.requires_grad,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _MXFP8LinearFunction.apply(input, self.weight, self.bias)


__all__ = ["MXFP8Linear"]
