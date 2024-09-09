from typing import Callable

import torch
from parameterized import parameterized

from khd import swiglu_torch, swiglu_triton

from .test_commons import TestCommons


class SwigluTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(), [torch.device("cuda")], TestCommons.get_dtypes()
        )
    )
    def test_swiglu_triton(self, size: tuple[int], device: torch.device, dtype: torch.dtype) -> None:
        self._test_swiglu(size, device, dtype, swiglu_triton)

    def _test_swiglu(self, size: int, device: torch.device, dtype: torch.dtype, function: Callable) -> None:
        x = torch.randn(size, device=device, dtype=dtype)
        y = torch.randn(size, device=device, dtype=dtype)

        z_kernel = function(x, y)
        z_expected = swiglu_torch(x, y)

        self.assert_equal_tensors(z_kernel, z_expected, True)
