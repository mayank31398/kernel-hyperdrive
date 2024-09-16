from functools import partial
from typing import Callable

import torch
from parameterized import parameterized

from khd import swiglu_torch, swiglu_triton

from .test_commons import TestCommons


class SwigluTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [
                partial(swiglu_torch, memory_efficient=True),
                partial(swiglu_triton, memory_efficient=False),
                partial(swiglu_triton, memory_efficient=True),
            ],
        )
    )
    def test_swiglu(self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y_kernel, y_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        x_kernel_non_leaf = self.get_non_leaf_tensor(x_kernel)
        x_expected_non_leaf = self.get_non_leaf_tensor(x_expected)
        y_kernel_non_leaf = self.get_non_leaf_tensor(y_kernel)
        y_expected_non_leaf = self.get_non_leaf_tensor(y_expected)

        z_kernel = function(x_kernel_non_leaf, y_kernel_non_leaf)
        z_expected = swiglu_torch(x_expected_non_leaf, y_expected_non_leaf, memory_efficient=False)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=5e-6, rtol_float32=0)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
        self.assert_equal_tensors(y_kernel.grad, y_expected.grad, False, atol_float32=5e-6, rtol_float32=0)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [4],
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [partial(swiglu_triton, memory_efficient=True)],
        )
    )
    def test_swiglu_memory_efficient_raises_error_with_leaf_tensors(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype, function: Callable
    ) -> None:
        x, y = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        with self.assertRaises(RuntimeError, msg="leaf variables can't be used in an in-place operation"):
            function(x, y)