# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "OverrideDefinitions",
]


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    timeout: int | None = None
    # Number of allocated GPUs that participate in the training world.
    train_ngpu: int | None = None
    # CLI option that receives the allocated non-training GPU ids.
    extra_gpu_arg: str | None = None

    def __post_init__(self) -> None:
        if self.train_ngpu is not None and not 1 <= self.train_ngpu <= self.ngpu:
            raise ValueError("train_ngpu must be between 1 and ngpu.")
        if self.extra_gpu_arg is not None and (
            self.train_ngpu is None or self.train_ngpu == self.ngpu
        ):
            raise ValueError(
                "extra_gpu_arg requires train_ngpu to leave at least one reserved "
                "GPU outside the training job."
            )

    def __repr__(self):
        return self.test_descr
