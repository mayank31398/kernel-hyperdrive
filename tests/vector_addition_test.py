from typing import Callable

import torch
from parameterized import parameterized

from kernels import vector_addition_cuda, vector_addition_pytorch, vector_addition_triton

from .test_commons import TestCommons


class VectorAdditionTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_1d_tensor_sizes(), [torch.device("cuda")], TestCommons.get_dtypes()
        )
    )
    def test_vector_addition_triton(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        self._test_vector_addition(size, device, dtype, vector_addition_triton)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_1d_tensor_sizes(), [torch.device("cuda")], TestCommons.get_dtypes()
        )
    )
    def test_vector_addition_cuda(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        self._test_vector_addition(size, device, dtype, vector_addition_cuda)

    def _test_vector_addition(self, size: int, device: torch.device, dtype: torch.dtype, function: Callable) -> None:
        x = torch.randn(size, device=device, dtype=dtype)
        y = torch.randn(size, device=device, dtype=dtype)

        z_kernel = function(x, y)
        z_expected = vector_addition_pytorch(x, y)

        self.assert_equal_tensors(z_kernel, z_expected, True)
