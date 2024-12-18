from torch.fx import Node
from torch.fx.graph_module import GraphModule

from ..kernels import swiglu_cute
from .swiglu_unchunked import _check_swiglu_after_chunk_and_get_output_node


def replace_swiglu(gm: GraphModule, node: Node) -> None:
    valid, output = _check_swiglu_after_chunk_and_get_output_node(*list(node.users.keys()))
    if not valid:
        return

    print("replacing with swiglu")

    with gm.graph.inserting_after(node):
        x = node.kwargs.get("input", node.args[0])
        new_node = gm.graph.call_function(swiglu_cute, args=(x,))

    node.replace_all_uses_with(new_node)
    output.replace_all_uses_with(new_node)

    gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
