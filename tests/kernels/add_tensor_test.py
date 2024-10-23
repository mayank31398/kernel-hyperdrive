from functools import partial
from typing import Callable

import torch
from parameterized import parameterized

from khd import KernelBackend, add_tensor_khd, add_tensor_torch

from ..test_commons import TestCommons


class AddTensorTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [
                partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vectorized_loop_size=1),
                partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vectorized_loop_size=2),
                partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vectorized_loop_size=4),
                torch.compile(partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vectorized_loop_size=1)),
                torch.compile(partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vectorized_loop_size=2)),
                torch.compile(partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vectorized_loop_size=4)),
                partial(add_tensor_khd, kernel_backend=KernelBackend.triton),
                torch.compile(partial(add_tensor_khd, kernel_backend=KernelBackend.triton)),
            ],
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            [torch.float16, torch.bfloat16],
            [
                partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vectorized_loop_size=8),
                torch.compile(partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vectorized_loop_size=8)),
            ],
        )
    )
    def test_add_tensor(self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y_kernel, y_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(x_kernel, y_kernel)
        z_expected = add_tensor_torch(x_expected, y_expected)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
        self.assert_equal_tensors(y_kernel.grad, y_expected.grad, True)
