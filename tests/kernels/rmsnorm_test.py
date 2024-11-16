from typing import Callable

import torch
from parameterized import parameterized

from khd import KernelBackend, rmsnorm_khd, rmsnorm_torch

from ..test_commons import TestCommons


_EPSILON = 1e-5


class RMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            # TestCommons.get_2d_tensor_sizes(),  # size
            [(2437, 2437)],
            [torch.device("cuda")],  # device
            TestCommons.get_dtypes(),  # dtype
            [True],  # memory_efficient
            [KernelBackend.triton],  # kernel_backend_forward
            [KernelBackend.triton],  # kernel_backend_backward
            [1],  # BLOCK_SIZE_B_forward
            [1],  # BLOCK_SIZE_B_backward
            [4096],  # BLOCK_SIZE_H_forward
            [4096],  # BLOCK_SIZE_H_backward
            [rmsnorm_khd, rmsnorm_khd],  # function
        )
    )
    def test_rmsnorm(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        memory_efficient: bool,
        kernel_backend_forward: KernelBackend,
        kernel_backend_backward: KernelBackend,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_H_backward: int,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        weight_kernel, weight_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)

        z_kernel = function(
            x=x_kernel,
            weight=weight_kernel,
            eps=_EPSILON,
            memory_efficient=memory_efficient,
            kernel_backend_forward=kernel_backend_forward,
            kernel_backend_backward=kernel_backend_backward,
            BLOCK_SIZE_B_forward=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_B_backward=BLOCK_SIZE_B_backward,
            BLOCK_SIZE_H_forward=BLOCK_SIZE_H_forward,
            BLOCK_SIZE_H_backward=BLOCK_SIZE_H_backward,
        )
        z_expected = rmsnorm_torch(x=x_expected, weight=weight_expected, eps=_EPSILON)

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False)
        print((z_kernel - z_expected).abs().max())
        print((x_kernel.grad - x_expected.grad).abs().max())
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
        # self.assert_equal_tensors(weight_kernel.grad, weight_expected.grad, True)
