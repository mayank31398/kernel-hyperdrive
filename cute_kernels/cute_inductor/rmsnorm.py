import torch
from torch.fx import Node
from torch.fx.graph_module import GraphModule

from ..kernels import rmsnorm_cute
from .constants import CALL_FUNCTION


def replace_rmsnorm(gm: GraphModule, node: Node) -> None:
    if node.op == CALL_FUNCTION and node.target == torch.rms_norm:
        # delete normalized_shape from the args (position 1)
        args = node.args[:1] + node.args[2:]
        kwargs = {key: value for key, value in node.kwargs.items()}

        # delete normalized_shape from the kwargs
        kwargs.pop("normalized_shape", None)

        # rename input to x
        input = kwargs.pop("input", None)
        if input is not None:
            kwargs["x"] = input

        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(rmsnorm_cute, args=tuple(args), kwargs=kwargs)

        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
