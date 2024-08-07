from typing import Callable

import torch
import torch.nn as nn

from ..torch_implementation import Experts_Torch, MoE_Torch
from .kernel import flatten_and_sort, group, group_bwd_W, padded_block_indices, scatter2scatter


class ParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):

        output = scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=k,
            x_grouped=grouped_in,
            y_grouped=grouped_out,
        )

        if gates is None:
            output_expanded = None
        else:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = torch.bmm(gates[:, None, :], output_expanded).squeeze(1)

        ctx.save_for_backward(
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            output_expanded,
        )

        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_out):
        (
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            output_expanded,
        ) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out

        if gates is None:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None
        else:
            # calculate gates gradient
            d_gates = torch.bmm(output_expanded, grad_out[:, :, None]).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            # print("expanded and grouping")
            grouped_grad_out = output_expanded.flatten(0, 1)  # reuse expanded buffer later

        if grouped_out:
            grouped_grad_out = grad_out
        else:
            grouped_grad_out = group(
                grad_out, sorted_scattered_idxs, fan_out=gate_fan, coeff=gates_flat, out=grouped_grad_out
            )

        if grouped_in:
            grouped_x = x
            d_expanded_input = None
        else:
            grouped_x = group(x, sorted_scattered_idxs, fan_out=k)
            d_expanded_input = grouped_x

        d_weights = group_bwd_W(
            DY=grouped_grad_out, X=grouped_x, expert_offsets=expert_offsets, E=expert_weights.size(0)
        )

        d_expanded_input = scatter2scatter(
            X=grouped_grad_out,
            x_grouped=True,
            W=expert_weights.permute(0, 2, 1),
            padded_block_idxs=padded_block_idxs,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=1,
            y_grouped=grouped_in,
            out=d_expanded_input,  # Reuse grouped_x buffer
        )

        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)

        # print("backward end.")
        return (
            # x, expert_weights, k,
            d_input,
            d_weights,
            None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None,
            None,
            # padded_block_idxs, expert_offsets,
            None,
            None,
            # gates
            d_gates,
            None,
            None,
        )


def scattered_experts(
    inputs,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    padded_block_idxs,
    expert_offsets,
    gates=None,
    grouped_in=False,
    grouped_out=False,
):
    results = ParallelLinear.apply(
        inputs,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates,
        grouped_in,
        grouped_out,
    )
    return results


class Experts_Triton(Experts_Torch):
    def forward(
        self,
        inputs,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        results = scattered_experts(
            inputs,
            self.weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )

        return results


class MoE_Triton(MoE_Torch):
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
        nn.Module.__init__(self)

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

        self.c_fc = Experts_Triton(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.act = activation_function

        self.c_proj = Experts_Triton(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

    def _compute_experts(
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = flatten_and_sort(selected_experts)
            padded_block_idxs, expert_offsets = padded_block_indices(sorted_expert_idxs, self.num_experts)

        hidden_states = self.c_fc(
            hidden_states,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_out=True,
        )
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(
            hidden_states,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_in=True,
            gates=router_weights,
        )
        return hidden_states
