from collections import defaultdict
from collections.abc import Collection

from texttable import Texttable

from chains.core.graph import Graph, Node
from chains.core.shape import Shape
from chains.tools import exportable_graph
from chains.tools.exportable_graph import (
    ExportGraph,
    ExportSubGraph,
    ExportEdge,
    ExportNodeStyle,
)


def export_sub(
    edges: Collection[ExportEdge],
    sub_graph: ExportSubGraph,
    namespace: tuple[str, ...],
    tt: Texttable,
):
    for n in sub_graph.nodes:
        outgoings = [e.label for e in edges if e.src == n.name]
        tt.add_row(
            [
                ".".join(namespace),
                n.name,
                _category(n.style),
                "" if len(outgoings) == 0 else outgoings[0],
            ]
        )

    for ns, sg in sub_graph.sub_groups.items():
        export_sub(edges, sg, namespace + (ns,), tt)


def _category(node: Node) -> str:
    style = exportable_graph._node_style(node)
    match style:
        case ExportNodeStyle.VARIABLE:
            return "variable"
        case ExportNodeStyle.PLACEHOLDER:
            return "placeholder"
        case _:
            return ""


def export(graph: Graph) -> str:
    tt = Texttable()
    tt.set_deco(Texttable.VLINES | Texttable.BORDER | Texttable.HEADER)
    tt.add_rows(
        [["namespace", "node", "category", "output shape", "#params"]], header=True
    )
    tt.set_cols_align(["l", "l", "l", "r", "r"])

    out_shapes: dict[str, Shape] = defaultdict()
    for node in graph._forward_path:
        in_shapes = [out_shapes[in_node.name] for in_node in node.incoming_nodes]
        out_shapes[node.name] = node.op.compute_out_shape(*in_shapes)

    total_params = 0
    for node in graph._forward_path:
        param_count = (
            out_shapes[node.name].size()
            if exportable_graph._node_style(node) == ExportNodeStyle.VARIABLE
            else 0
        )
        total_params += param_count

        tt.add_row(
            [
                node.namespace,
                node.name,
                _category(node),
                out_shapes[node.name],
                (f"{param_count:,d}" if param_count else ""),
            ]
        )

    # graph.root
    # tp =
    # Total
    # params: 25450
    # Trainable
    # params: 25450
    # Non - trainable
    # params: 0

    summary = f"{total_params:,d}" if total_params else ""
    return tt.draw() + f"\nTotal params: {summary}\n"
