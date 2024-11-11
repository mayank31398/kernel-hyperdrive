from typing import Callable

import torch
from parameterized import parameterized

from khd import KernelBackend, add_scalar_khd, add_scalar_torch

from ..test_commons import TestCommons


class AddTensorTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [KernelBackend.cuda],
            [1, 2, 4],
            [1024],
            [add_scalar_khd, torch.compile(add_scalar_khd)],
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            [torch.bfloat16, torch.float16],
            [KernelBackend.cuda],
            [8],
            [1024],
            [add_scalar_khd, torch.compile(add_scalar_khd)],
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [KernelBackend.triton],
            [None],
            [1024],
            [add_scalar_khd, torch.compile(add_scalar_khd)],
        )
    )
    def test_add_tensor(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        kernel_backend: KernelBackend,
        vector_instruction_width: int,
        BLOCK_SIZE: int,
        function: Callable,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y = 0.42

        z_kernel = function(
            x_kernel,
            y,
            kernel_backend=kernel_backend,
            vector_instruction_width=vector_instruction_width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        z_expected = add_scalar_torch(x_expected, y)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
