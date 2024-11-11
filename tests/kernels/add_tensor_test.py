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
            [KernelBackend.cuda],
            [1, 2, 4],
            [1024],
            [False, True],
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            [torch.bfloat16, torch.float16],
            [KernelBackend.cuda],
            [8],
            [1024],
            [False, True],
        )
        + TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [KernelBackend.triton],
            [None],
            [1024],
            [False, True],
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
        torch_compile: bool,
    ) -> None:
        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y_kernel, y_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        if torch_compile:
            z_kernel = torch.compile(add_tensor_khd)(x_kernel, y_kernel)
        else:
            z_kernel = add_tensor_khd(x_kernel, y_kernel)

        z_expected = add_tensor_torch(x_expected, y_expected)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
        self.assert_equal_tensors(y_kernel.grad, y_expected.grad, True)
