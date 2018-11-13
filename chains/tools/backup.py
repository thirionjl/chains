import h5py

import chains.core.saver as saver
from chains.core.graph import Graph
from chains.front.network import Network


def save_weights(g: Graph, file_name: str):
    with h5py.File(file_name, "w") as f:
        _save_weights_into(g, f)


def _save_weights_into(g: Graph, f: h5py.Group):
    for n in g.all_nodes:
        values = saver.save(n.op)
        if values is not None:
            if n.name not in f:
                grp = f.create_group(n.name)
            else:
                grp = f[n.name]
            for name, data in values.items():
                grp.create_dataset(name=name, dtype=str(n.dtype), data=data)


def restore_weights(g: Graph, file_name: str):
    with h5py.File(file_name, "r") as f:
        _restore_weights_into(g, f)


def _restore_weights_into(g: Graph, f: h5py.Group):
    for n in g.all_nodes:
        if n.name in f:
            node = f[n.name]
            to_restore = {}
            for key, hdf5_value in node.items():
                if hasattr(hdf5_value, "value"):
                    to_restore[key] = hdf5_value.value
            saver.restore(n.op, to_restore)


def save_network(n: Network, file_name: str):
    save_weights(n.cost_graph, file_name)


def restore_network(n: Network, file_name: str):
    restore_weights(n.cost_graph, file_name)
