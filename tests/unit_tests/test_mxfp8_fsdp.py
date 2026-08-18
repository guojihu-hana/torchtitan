# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels.mxfp8")

import torchtitan.components.quantization.mxfp8_fsdp as mxfp8_fsdp  # noqa: E402
from torchtitan.components.quantization.mxfp8_fsdp import (  # noqa: E402
    MXFP8FSDPComputeWeight,
    MXFP8FSDPWeight,
)
from torchtitan.components.quantization.mxfp8_linear import MXFP8Linear  # noqa: E402


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _get_weight_param(linear):
    state = fully_shard.state(linear)
    param_group = state._fsdp_param_group
    assert param_group is not None
    return next(
        param
        for param in param_group.fsdp_params
        if param._module_info.param_name == "weight"
    )


def _run_reshard_after_forward(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        linear = (
            MXFP8Linear.Config(
                in_features=128,
                out_features=128,
                bias=False,
            )
            .build()
            .cuda()
            .bfloat16()
        )
        linear.compile()
        fully_shard(
            linear,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=True,
        )
        assert isinstance(linear.weight.to_local(), MXFP8FSDPWeight)

        input_MK = torch.randn(
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output_MN = linear(input_MK)
        weight_param = _get_weight_param(linear)
        inner_tensor_ids = tuple(map(id, weight_param._unsharded_inner_tensors))
        assert isinstance(linear.weight.to_local(), MXFP8FSDPWeight)
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        output_MN.sum().backward()
        assert isinstance(linear.weight.to_local(), MXFP8FSDPWeight)
        assert tuple(map(id, weight_param._unsharded_inner_tensors)) == inner_tensor_ids
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )
    finally:
        dist.destroy_process_group()


def _run_pp_cache_lifecycle(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    original_quantize_weight = mxfp8_fsdp._quantize_weight_32x32
    num_quantize_calls = 0

    def counted_quantize_weight(weight_NK: torch.Tensor):
        nonlocal num_quantize_calls
        num_quantize_calls += 1
        return original_quantize_weight(weight_NK)

    mxfp8_fsdp._quantize_weight_32x32 = counted_quantize_weight
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        linear = (
            MXFP8Linear.Config(
                in_features=128,
                out_features=128,
                bias=False,
            )
            .build()
            .cuda()
            .bfloat16()
        )
        fully_shard(
            linear,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=False,
        )
        linear.set_is_last_backward(False)
        linear.set_reshard_after_backward(False)
        linear.set_requires_gradient_sync(False)

        inputs = [
            torch.randn(
                64,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(2)
        ]
        outputs = [linear(input_MK) for input_MK in inputs]
        assert num_quantize_calls == 1
        weight_param = _get_weight_param(linear)
        assert isinstance(linear.weight, MXFP8FSDPComputeWeight)
        assert len(weight_param._unsharded_inner_tensors) == 3
        assert (
            linear.weight.q_weight_dgrad_NK.data_ptr()
            == linear.weight.q_weight_fprop_KN.data_ptr()
        )
        inner_tensor_ids = tuple(map(id, weight_param._unsharded_inner_tensors))
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        outputs[0].sum().backward()
        assert num_quantize_calls == 1
        assert isinstance(linear.weight, MXFP8FSDPComputeWeight)

        linear.set_is_last_backward(True)
        linear.set_reshard_after_backward(True)
        linear.set_requires_gradient_sync(True)
        outputs[1].sum().backward()
        assert num_quantize_calls == 1
        assert isinstance(linear.weight.to_local(), MXFP8FSDPWeight)
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        output_MN = linear(inputs[0].detach())
        assert num_quantize_calls == 2
        assert isinstance(linear.weight, MXFP8FSDPComputeWeight)
        assert tuple(map(id, weight_param._unsharded_inner_tensors)) == inner_tensor_ids
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param.all_gather_outputs
        )
        output_MN.sum().backward()
    finally:
        mxfp8_fsdp._quantize_weight_32x32 = original_quantize_weight
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
    reason="MXFP8 requires SM100 or later",
)
@pytest.mark.parametrize(
    "target",
    [_run_reshard_after_forward, _run_pp_cache_lifecycle],
    ids=["reshard-after-forward", "pp-cache"],
)
def test_mxfp8_fsdp_weight_lifecycle(target):
    mp.spawn(
        target,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )
