from functools import partial
from typing import Callable

import torch
from parameterized import parameterized

from khd import vector_addition_cuda, vector_addition_torch, vector_addition_triton

from .test_commons import TestCommons


class VectorAdditionTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [
                partial(vector_addition_cuda, in_place=False),
                partial(vector_addition_cuda, in_place=True),
                partial(vector_addition_triton, in_place=False),
                partial(vector_addition_triton, in_place=True),
            ],
        )
    )
    def test_vector_addition(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y_kernel, y_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(x_kernel, y_kernel)
        z_expected = vector_addition_torch(x_expected, y_expected, in_place=False)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
        self.assert_equal_tensors(y_kernel.grad, y_expected.grad, True)
