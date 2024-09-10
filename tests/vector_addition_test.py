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
            [vector_addition_cuda, vector_addition_triton],
        )
    )
    def test_vector_addition(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable
    ) -> None:
        x = torch.randn(size, device=device, dtype=dtype)
        y = torch.randn(size, device=device, dtype=dtype)

        z_kernel = function(x, y)
        z_expected = vector_addition_torch(x, y)

        self.assert_equal_tensors(z_kernel, z_expected, True)
