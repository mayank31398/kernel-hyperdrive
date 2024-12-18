import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.graph_module import GraphModule

from ..kernels import swiglu_unchunked_cute
from .constants import CALL_METHOD


def replace_swiglu_unchunked(gm: GraphModule, node: Node) -> None:
    if node.op == CALL_METHOD and node.target == torch.chunk.__name__:
        print(node.args, node.kwargs)
        # if len(node.args) == 2 and node.args[1] == 2:
        #     with gm.graph.inserting_after(node):
        #         # Create a new node for the custom chunk_silu function
        #         new_node = gm.graph.call_function(chunk_silu, args=(node.args[0],))

        #     # Replace all uses of the old node with the new node
        #     node.replace_all_uses_with(new_node)
        #     gm.graph.erase_node(node)

        # node.replace_all_uses_with(new_node)
        # gm.graph.erase_node(node)
