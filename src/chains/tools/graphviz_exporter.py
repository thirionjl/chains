import tempfile
from pathlib import Path
from typing import Optional

import graphviz
import graphviz.dot

from chains.core.graph import Graph
from chains.tools import exportable_graph
from chains.tools.exportable_graph import (
    ExportNodeStyle,
    ExportSubGraph,
)
from chains.utils import validate

element_styles = {
    ExportNodeStyle.VARIABLE: {"fillcolor": "purple:pink"},
    ExportNodeStyle.PLACEHOLDER: {"fillcolor": "crimson:pink"},
    ExportNodeStyle.REGULARIZATION: {"fillcolor": "goldenrod", "penwidth": "0.2"},
    ExportNodeStyle.OTHER: {"fillcolor": "blue:cornflowerblue"},
}

edge_base_style = {
    "fontsize": "7pt",
    "color": "azure4",
    "fontcolor": "grey",
    "arrowsize": "0.4",
    "penwidth": "0.5",
}

node_base_style = {
    "style": "radial",
    "color": "white",
    "fontcolor": "white",
    "shape": "ellipse",
    "fontsize": "8pt",
    "penwidth": "0.5",
}


# Add clusters first
def export_nodes(digraph: graphviz.Digraph, grp_name: str, grp: ExportSubGraph) -> None:
    digraph.attr("edge", **edge_base_style)
    digraph.attr("node", **node_base_style)
    digraph.attr(label=grp_name)
    for gn in grp.nodes:
        digraph.node(gn.name, **element_styles[gn.style])
    for sub_grp_name, sub_grp in grp.sub_groups.items():
        with digraph.subgraph(name="cluster_" + sub_grp_name) as c:
            export_nodes(c, sub_grp_name, sub_grp)


def export(g: Graph, filepath: str = None, name: str = None) -> graphviz.Digraph:
    filename: Optional[str] = None
    if filename is None:
        with tempfile.NamedTemporaryFile() as fp:
            filename = fp.name
    elif filepath is not None:
        validate.is_not_blank("filepath", filepath)
        filename = str(Path(filepath).with_suffix(".gv"))

    graph = exportable_graph.get_exportable_graph(g)

    gvz = graphviz.Digraph(name=name, filename=filename, format="svg")
    gvz.attr(rankdir="LR")
    gvz.attr("edge", **edge_base_style)
    gvz.attr("node", **node_base_style)

    export_nodes(gvz, name, graph.root)

    for e in graph.edges:
        gvz.edge(e.src, e.dst, label=e.label)

    return gvz
