from functools import partial
from typing import Callable

import torch
from parameterized import parameterized

from khd import vector_addition_cuda, vector_addition_torch, vector_addition_triton

from .test_commons import TestCommons


class VectorAdditionTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_1d_tensor_sizes(), [torch.device("cuda")], TestCommons.get_dtypes(), [False, True]
        )
    )
    def test_vector_addition_cuda(self, size: int, device: torch.device, dtype: torch.dtype, in_place: bool) -> None:
        self._test_vector_addition(size, device, dtype, partial(vector_addition_cuda, in_place=in_place))

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_1d_tensor_sizes(), [torch.device("cuda")], TestCommons.get_dtypes(), [False, True]
        )
    )
    def test_vector_addition_triton(self, size: int, device: torch.device, dtype: torch.dtype, in_place: bool) -> None:
        self._test_vector_addition(size, device, dtype, partial(vector_addition_cuda, in_place=in_place))

    def _test_vector_addition(self, size: int, device: torch.device, dtype: torch.dtype, function: Callable) -> None:
        x = torch.randn(size, device=device, dtype=dtype)
        y = torch.randn(size, device=device, dtype=dtype)

        z_expected = vector_addition_torch(x, y, in_place=False)
        z_kernel = function(x, y)

        self.assert_equal_tensors(z_kernel, z_expected, True)
