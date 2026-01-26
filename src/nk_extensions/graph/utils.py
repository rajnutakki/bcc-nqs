import numpy as np
from data.utils import min_distance
from netket.graph import AbstractGraph
import itertools


def equivalent_distances(r1, r2, pbc_vecs, d_target):
    ds = []
    ndim = len(r1)
    for shift in itertools.product([-1, 0, 1], repeat=ndim):
        pbc_shift = np.array(shift) @ pbc_vecs
        d = np.linalg.norm(r1 - r2 + pbc_shift)
        if np.isclose(d, d_target, atol=1e-5):
            ds.append(d)

    return ds


def edges_from_graph_positions(graph: AbstractGraph) -> list[tuple[int, int, int]]:
    """
    Given a graph, return the colored edges of the graph, by using the positions of the nodes.
    Will take into account if there are repeated edges of same distance
    """
    edges = []
    pbc_vecs = np.array(
        [graph._pbc[i] * graph._lattice_dims[i] for i in range(graph.ndim)]
    )
    for i, j, c in graph.edges(return_color=True):
        d_target = min_distance(graph.positions[i], graph.positions[j], pbc_vecs)
        n_equivalent_edges = len(
            equivalent_distances(
                graph.positions[i], graph.positions[j], pbc_vecs, d_target
            )
        )
        edges += n_equivalent_edges * [(i, j, c)]

    return edges
