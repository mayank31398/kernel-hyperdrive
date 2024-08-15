import torch
import torch.nn as nn
from parameterized import parameterized
from transformers import set_seed

from khd import MoE_Torch, MoE_Triton

from .test_commons import TestCommons


SEED = 42


class ScatterMoETest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [2, 4, 6, 8],  # num_experts
            [2, 4],  # num_experts_per_tok
            [2048],  # hidden_size
            [8192],  # intermediate_size
            [True, False],  # is_glu
        )
    )
    def test_scattermoe_triton(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        is_glu: bool,
    ) -> None:
        self._test_scattermoe(
            device=device,
            dtype=dtype,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            is_glu=is_glu,
            module_class=MoE_Triton,
        )

    def _test_scattermoe(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        is_glu: bool,
        module_class: type[nn.Module],
    ) -> None:
        set_seed(SEED)

        if num_experts_per_tok > num_experts:
            self.skipTest(
                f"skipping test since number of experts per token ({num_experts_per_tok}) is more than number of experts ({num_experts})"
            )

        with torch.device(device):
            moe = module_class(
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_function=self.get_activation_function(is_glu=is_glu),
                is_glu=is_glu,
                add_bias=False,
                std=0.02,
            ).to(dtype=dtype)

            moe_torch = MoE_Torch(
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_function=self.get_activation_function(is_glu=is_glu),
                is_glu=is_glu,
                add_bias=False,
                std=0.02,
            ).to(dtype=dtype)

        moe_torch.load_state_dict(moe.state_dict())

        x = torch.randn(hidden_size, device=device, dtype=dtype)

        y = moe(x)[0]
        y_expected = moe_torch(x)[0]

        self.assert_equal_tensors(
            y,
            y_expected,
            False,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=2e-2,
            rtol_bfloat16=0,
            atol_float32=6e-3,
            rtol_float32=0,
        )
