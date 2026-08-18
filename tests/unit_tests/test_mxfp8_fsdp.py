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

from torchao.prototype.moe_training.mxfp8_fsdp import (  # noqa: E402
    MXFP8FSDPComputeWeight,
    MXFP8FSDPWeight,
)

from torchtitan.components.quantization.mxfp8_linear import MXFP8Linear  # noqa: E402


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _run_fsdp_cache_lifecycle(rank: int, world_size: int, port: int) -> None:
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
        fully_shard(
            linear,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=False,
        )

        assert isinstance(linear.weight.to_local(), MXFP8FSDPWeight)
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
        assert isinstance(linear.weight, MXFP8FSDPComputeWeight)

        state = fully_shard.state(linear)
        param_group = state._fsdp_param_group
        assert param_group is not None
        weight_param = next(
            param
            for param in param_group.fsdp_params
            if param._module_info.param_name == "weight"
        )
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param.all_gather_outputs
        )
        inner_tensor_ids = tuple(map(id, weight_param._unsharded_inner_tensors))
        assert all(
            tensor.untyped_storage().size() > 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        outputs[0].sum().backward()
        assert isinstance(linear.weight, MXFP8FSDPComputeWeight)

        linear.set_is_last_backward(True)
        linear.set_reshard_after_backward(True)
        linear.set_requires_gradient_sync(True)
        outputs[1].sum().backward()
        assert isinstance(linear.weight.to_local(), MXFP8FSDPWeight)
        assert all(
            tensor.untyped_storage().size() == 0
            for tensor in weight_param._unsharded_inner_tensors
        )

        optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output = linear(inputs[0].detach())
        assert isinstance(linear.weight, MXFP8FSDPComputeWeight)
        assert tuple(map(id, weight_param._unsharded_inner_tensors)) == inner_tensor_ids
        output.sum().backward()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
    reason="MXFP8 requires SM100 or later",
)
def test_mxfp8_fsdp_cache_lifecycle():
    mp.spawn(
        _run_fsdp_cache_lifecycle,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )
