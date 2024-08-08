import torch
import torch.nn as nn
from parameterized import parameterized

from kernel_hyperdrive import MoE_Torch, MoE_Triton

from .test_commons import TestCommons


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
        moe = module_class(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=self.get_activation_function(is_glu=is_glu),
            is_glu=is_glu,
            add_bias=False,
            std=0.02,
        )

        moe_torch = MoE_Torch(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=self.get_activation_function(is_glu=is_glu),
            is_glu=is_glu,
            add_bias=False,
            std=0.02,
        )

        moe_torch.load_state_dict(moe.state_dict())

        x = torch.randn(hidden_size, device=device, dtype=dtype)

        y = moe(x)
        y_expected = moe_torch(x)

        self.assert_equal_tensors(y, y_expected, True)
