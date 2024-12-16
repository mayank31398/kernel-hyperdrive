from itertools import product
from typing import Any
from unittest import TestCase

import torch
import torch.nn as nn
from torch.testing import assert_close

from cute_kernels import get_1d_tensor_sizes, get_2d_tensor_sizes, get_all_devices, get_dtypes, init_inductor


class TestCommons(TestCase):
    def setUp(self) -> None:
        return init_inductor(cache_size_limit=512)

    @staticmethod
    def get_all_devices() -> list[torch.device]:
        return get_all_devices()

    @staticmethod
    def get_dtypes() -> list[torch.dtype]:
        return get_dtypes()

    @staticmethod
    def get_1d_tensor_sizes() -> list[tuple[int]]:
        return get_1d_tensor_sizes()

    @staticmethod
    def get_2d_tensor_sizes(static_batch_size: int | None = None) -> list[tuple[int]]:
        return get_2d_tensor_sizes(static_batch_size)

    def make_args_matrix(*args_lists) -> list[Any]:
        return [p for p in product(*args_lists)]

    def assert_equal_tensors(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        exact_match: bool,
        rtol_float32: float = None,
        atol_float32: float = None,
        rtol_float16: float = None,
        atol_float16: float = None,
        rtol_bfloat16: float = None,
        atol_bfloat16: float = None,
    ) -> None:
        if exact_match:
            assert x.equal(y)
        else:
            assert x.dtype == y.dtype
            dtype = x.dtype

            if dtype == torch.float32:
                assert_close(x, y, rtol=rtol_float32, atol=atol_float32)
            elif dtype == torch.float16:
                assert_close(x, y, rtol=rtol_float16, atol=atol_float16)
            elif dtype == torch.bfloat16:
                assert_close(x, y, rtol=rtol_bfloat16, atol=atol_bfloat16)
            else:
                raise ValueError(f"unexpected dtype ({dtype})")

    def get_activation_function(self, is_glu: bool) -> nn.Module:
        return nn.GLU() if is_glu else nn.GELU(approximate="tanh")

    def get_random_duplicated_tensors(
        self, size: tuple[int], device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor]:
        x = torch.randn(size, device=device, dtype=dtype, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_()

        return x, x_clone
