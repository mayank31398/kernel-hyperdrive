from typing import Callable

import torch
from parameterized import parameterized

from khd import embedding_torch, embedding_triton

from ..test_commons import TestCommons


class EmbeddingTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(51, 17), (19, 239), (7, 7537), (9, 1749)],
            [(7153, 937), (27153, 1937), (97153, 2937), (17153, 31937)],
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [embedding_triton, torch.compile(embedding_triton)],
        )
    )
    def test_add_tensor(
        self,
        input_ids_size: tuple[int],
        wte_size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        function: Callable,
    ) -> None:
        vocab_size = wte_size[0] - 1
        input_ids = torch.randint(0, vocab_size, input_ids_size, device=device, dtype=torch.long)

        wte_kernel, wte_expected = self.get_random_duplicated_tensors(wte_size, device=device, dtype=dtype)

        z_kernel = function(input_ids, wte_kernel)
        z_expected = embedding_torch(input_ids, wte_expected)

        # z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(z_kernel, z_expected, True)
        # self.assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
        # self.assert_equal_tensors(y_kernel.grad, y_expected.grad, True)
