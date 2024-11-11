from functools import partial
from typing import Callable

import torch
from parameterized import parameterized

from khd import KernelBackend, rmsnorm_khd, rmsnorm_torch

from ..test_commons import TestCommons


_EPSILON = 1e-5


class RMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.triton],  # kernel_backend
            [32],  # BLOCK_SIZE_B
            [32],  # BLOCK_SIZE_H
            [rmsnorm_khd, torch.compile(rmsnorm_khd)],  # function
        )
    )
    def test_rmsnorm(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_B: int,
        BLOCK_SIZE_H: int,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)

        z_kernel = function(
            x=x_kernel,
            weight=weight_kernel,
            eps=_EPSILON,
            kernel_backend=kernel_backend,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
        z_expected = rmsnorm_torch(x=x_expected, weight=weight_expected, eps=_EPSILON)

        # z_kernel.mean().backward()
        # z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False)
        # self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
        # self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, True)
