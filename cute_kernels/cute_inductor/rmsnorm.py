import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.graph_module import GraphModule

from ..kernels import rmsnorm_cute
from .constants import CALL_FUNCTION


def replace_rmsnorm(gm: GraphModule, node: Node) -> None:
    if node.op == CALL_FUNCTION and node.target in [torch.rms_norm, F.rms_norm]:
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(rmsnorm_cute, args=(node.args[0], node.args[2], node.args[3]))

        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
