import os
from typing import Callable

import torch
import torch.nn.functional as F
from torch._dynamo import lookup_backend


_DEBUG_CUTEINDUCTOR = bool(os.getenv("DEBUG_CUTEINDUCTOR", 0))
_CALL_FUNCTION = "call_function"


class CuteInductor:
    def __init__(self, use_inductor: bool = True) -> None:
        self.use_inductor = use_inductor

    def compiler(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        from cute_kernels import rmsnorm_cute

        if _DEBUG_CUTEINDUCTOR:
            print("graph before cute inductor")
            gm.print_readable()

        for node in gm.graph.nodes:
            if node.op == _CALL_FUNCTION:
                # RMSNorm
                if node.target in [torch.rms_norm, F.rms_norm]:
                    with gm.graph.inserting_after(node):
                        new_node = gm.graph.call_function(
                            rmsnorm_cute, args=(node.args[0], node.args[2], node.args[3])
                        )
                    node.replace_all_uses_with(new_node)
                    gm.print_readable()
                    gm.graph.erase_node(node)

        if _DEBUG_CUTEINDUCTOR:
            print("graph after cute inductor")
            gm.print_readable()

        if self.use_inductor:
            inductor = lookup_backend("inductor")
            compiled = inductor(gm, example_inputs)
        else:
            compiled = gm.forward

        return compiled
