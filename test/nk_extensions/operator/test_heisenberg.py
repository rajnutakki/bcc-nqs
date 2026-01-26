import pytest
import netket as nk
import nk_extensions as nke
import numpy as np

# For counting number of edges
graphs = (
    pytest.param(
        nk.graph.Chain(length=10, pbc=True, max_neighbor_order=3),
        10 * 3 * 2 / 2,  # length*coordination_number/2 for each coupling
        id="Chain",
    ),
    pytest.param(
        nk.graph.Square(length=6, pbc=True, max_neighbor_order=2),
        36 * 4 / 2 + 36 * 4 / 2,
        id="Square",
    ),
    pytest.param(
        nke.graph.BCC_cubic(extent=(4, 2, 2), pbc=True, max_neighbor_order=2),
        32 * 8 / 2 + 32 * 6 / 2,
        id="BCC_cubic",
    ),
)


@pytest.mark.parametrize("graph, expected", graphs)
def test_num_edges(graph, expected):
    edges = nke.graph.utils.edges_from_graph_positions(graph)
    assert len(edges) == expected


# For counting number of edges
graphs = (
    pytest.param(nk.graph.Chain(length=10, pbc=True, max_neighbor_order=3), id="Chain"),
    pytest.param(
        nk.graph.Square(length=4, pbc=True, max_neighbor_order=2), id="Square"
    ),
)


@pytest.mark.parametrize("graph", graphs)
def test_heisenberg_hamiltonians(graph):
    J = [0.1 * i for i in range(graph._max_neighbor_order)]
    hilbert = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
    ham1 = nk.operator.Heisenberg(hilbert, graph, J=J)
    edges = nke.graph.utils.edges_from_graph_positions(graph)
    ham2 = nke.operator.heisenberg.heisenberg_edges(hilbert, edges, J=J)
    e1 = nk.exact.lanczos_ed(ham1)[0]
    e2 = nk.exact.lanczos_ed(ham2)[0]
    assert np.isclose(e1, e2)
