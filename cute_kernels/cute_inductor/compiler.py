import os
from typing import Callable

import torch
from torch._dynamo import lookup_backend

from ..utils import get_boolean_env_variable
from .rmsnorm import replace_rmsnorm


_DEBUG_CUTEINDUCTOR = get_boolean_env_variable("DEBUG_CUTEINDUCTOR", False)


class CuteInductor:
    def __init__(self, use_inductor: bool = True) -> None:
        self.use_inductor = use_inductor

    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        if _DEBUG_CUTEINDUCTOR:
            print("graph before cute inductor")
            gm.print_readable()

        for node in gm.graph.nodes:
            replace_rmsnorm(gm, node)

        if _DEBUG_CUTEINDUCTOR:
            print("graph after cute inductor")
            gm.print_readable()

        if self.use_inductor:
            inductor = lookup_backend("inductor")
            compiled = inductor(gm, example_inputs)
        else:
            compiled = gm.forward

        return compiled
