import pytest
import nk_extensions as nke


@pytest.mark.parametrize(
    "extent, max_neighbor_order, tetragonal_distortion",
    [
        ((4, 4, 4), 2, 1.0),
        ((6, 4, 4), 2, 1.0),
        ((4, 4, 4), 3, 0.985),
        ((6, 4, 4), 3, 0.985),
    ],
)
def test_cubic(extent, max_neighbor_order, tetragonal_distortion):
    graph = nke.graph.BCC_cubic(
        extent=extent,
        pbc=True,
        max_neighbor_order=max_neighbor_order,
        tetragonal_distortion=tetragonal_distortion,
    )
    if tetragonal_distortion == 1.0:
        expected_edge_count = (4 * graph.n_nodes, 3 * graph.n_nodes)  # J1, J2
    else:
        expected_edge_count = (
            4 * graph.n_nodes,
            graph.n_nodes,
            2 * graph.n_nodes,
        )  # J1, Jc, Jab

    for color in range(max_neighbor_order):
        edges = graph.edges(filter_color=color)
        assert len(edges) == expected_edge_count[color], (
            f"Unexpected number of edges for color {color}, got {len(edges)}, expected {expected_edge_count[color]}"
        )
