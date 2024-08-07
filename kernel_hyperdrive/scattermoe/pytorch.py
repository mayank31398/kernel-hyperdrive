from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Experts_PyTorch(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, add_bias: bool = True, std: float | None = None
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def forward(self, input: torch.Tensor, num_experts_per_token: torch.Tensor) -> torch.Tensor:
        input = input.split(num_experts_per_token.tolist(), dim=0)
        input = [
            F.linear(input[i], self.weight[i], None if self.bias is None else self.bias[i])
            for i in range(self.num_experts)
        ]
        input = torch.cat(input, dim=0)
        return input

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.zero_()


class MoE_PyTorch(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: Callable,
        is_glu: bool,
        add_bias: bool,
        std: float,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate = nn.Linear(
            in_features=self.hidden_size,
            out_features=num_experts,
            bias=False,
            std=std,
        )

        self.c_fc = Experts_PyTorch(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.act = activation_function

        self.c_proj = Experts_PyTorch(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)
        hidden_states = self._compute_experts(hidden_states, router_weights, selected_experts)

        hidden_states = hidden_states.view(original_shape)

        return hidden_states, router_logits

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)
        router_weights = F.softmax(router_weights.float(), dim=-1)

        # we cast back to the input dtype
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        total_q = hidden_states.shape[0]

        batch_index, batch_gates, num_experts_per_token = self._compute_expert_assignment(
            router_weights, selected_experts
        )

        expert_inputs = hidden_states[batch_index]

        hidden_states = self.c_fc(expert_inputs, num_experts_per_token)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states, num_experts_per_token)

        hidden_states = hidden_states * batch_gates.unsqueeze(-1)  # [:, None]
        zeros = torch.zeros((total_q, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = zeros.index_add(0, batch_index, hidden_states)

        return hidden_states

    def _compute_expert_assignment(
        self, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> tuple[torch.Tensor]:
        selected_experts = selected_experts.flatten()

        num_experts_per_token = selected_experts.bincount(minlength=self.num_experts)

        # sort and group input tokens according to expert assignment
        _, index_sorted_experts = selected_experts.sort(0)  # [num_tokens * top_k]
        batch_index = index_sorted_experts // self.top_k  # [num_tokens * top_k]

        # gather the gate values for grouped input tokens
        router_weights = router_weights.flatten()  # [num_tokens * top_k]
        batch_gates = router_weights[index_sorted_experts]  # [num_tokens * top_k]

        return batch_index, batch_gates, num_experts_per_token

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices
