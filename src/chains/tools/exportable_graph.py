from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

from chains.core.graph import Graph, Node
from chains.core.ops import Var, Constant, Placeholder
from chains.core.ops_regularization import Dropout, L2NormRegularization
from chains.core.shape import Shape


class ExportNodeStyle(Enum):
    VARIABLE = auto()
    PLACEHOLDER = auto()
    REGULARIZATION = auto()
    OTHER = auto()


@dataclass(frozen=True)
class ExportEdge:
    src: str
    dst: str
    label: str
    style: ExportNodeStyle


@dataclass(frozen=True)
class ExportNode:
    name: str
    style: ExportNodeStyle


@dataclass
class ExportSubGraph:
    nodes: set[ExportNode] = field(default_factory=set)
    sub_groups: dict[str, "ExportSubGraph"] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportGraph:
    root: ExportSubGraph
    edges: set[ExportEdge]


NODE_STYLES = {
    ExportNodeStyle.REGULARIZATION: [Dropout, L2NormRegularization],
    ExportNodeStyle.VARIABLE: [Var, Constant],
    ExportNodeStyle.PLACEHOLDER: [Placeholder],
}

INVERTED_NODE_STYLES: dict[type, ExportNodeStyle] = dict()
for k, values in NODE_STYLES.items():
    for v in values:
        INVERTED_NODE_STYLES[v] = k


def _node_style(n: Node) -> ExportNodeStyle:
    return INVERTED_NODE_STYLES.get(n.op.__class__, ExportNodeStyle.OTHER)


def _add_node_to_export_graph(root: ExportSubGraph, node: Node):
    parent = root
    if node.namespace:
        for item in node.namespace.split("."):
            if item not in parent.sub_groups:
                parent.sub_groups[item] = ExportSubGraph()
            parent = parent.sub_groups[item]
    parent.nodes.add(ExportNode(node.name, _node_style(node)))


def get_exportable_graph(graph: Graph) -> ExportGraph:
    # compute nodes_to_draw
    tree: ExportSubGraph = ExportSubGraph()
    out_shapes: dict[str, Shape] = defaultdict()
    # Todo Extract or expose or move method inside graph?
    for node in graph._forward_path:
        in_shapes = [out_shapes[in_node.name] for in_node in node.incoming_nodes]
        out_shapes[node.name] = node.op.compute_out_shape(*in_shapes)
        _add_node_to_export_graph(tree, node)

    # compute edges
    edges_to_draw: set[ExportEdge] = set()
    for dst_node in graph.all_nodes:
        for src_node in dst_node.incoming_nodes:
            dims = out_shapes[src_node.name]
            edges_to_draw.add(
                ExportEdge(
                    src=src_node.name,
                    dst=dst_node.name,
                    label="scalar" if dims.is_scalar() else str(dims),
                    style=_node_style(dst_node),
                )
            )

    return ExportGraph(tree, edges_to_draw)
