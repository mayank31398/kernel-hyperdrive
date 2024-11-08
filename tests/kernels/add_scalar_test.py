from functools import partial
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
            [
                partial(add_scalar_khd, kernel_backend=KernelBackend.cuda, BLOCK_SIZE=1024),
                partial(add_scalar_khd, kernel_backend=KernelBackend.triton, BLOCK_SIZE=1024),
                torch.compile(partial(add_scalar_khd, kernel_backend=KernelBackend.cuda, BLOCK_SIZE=1024)),
                torch.compile(partial(add_scalar_khd, kernel_backend=KernelBackend.triton, BLOCK_SIZE=1024)),
            ],
        )
    )
    def test_add_tensor(self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y = 0.42

        z_kernel = function(x_kernel, y)
        z_expected = add_scalar_torch(x_expected, y)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
