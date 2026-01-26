from netket.utils.group._point_group import PointGroup
from netket.graph import AbstractGraph
from netket.graph.lattice import InvalidSiteError
from nk_extensions.group.utils import is_symmetric


def extract_valid_point_group(
    graph: AbstractGraph, point_group: PointGroup
) -> PointGroup:
    """
    Given a graph and a point group, check which elements of the point group are a valid symmetry of the graph
    by trying graph.point_group(element) for each element of the point group.
    First checks if the sites are mapped to valid sites, then checks the set of edges are invariant under the symmetry operation
    Return:
        A PointGroup object containing the subset of elements of point_group that are valid symmetries of the graph.
    """
    # Check if point group element maps sites to valid sites
    symm_elems = []
    for elem in point_group:
        group = PointGroup([elem], ndim=graph.ndim)
        try:
            graph.point_group(group)
            symm_elems.append(elem)
        except InvalidSiteError:
            pass

    # Check if these elements also leave the edges invariant
    initial_pg = graph.point_group(PointGroup(symm_elems, ndim=graph.ndim))
    final_pg_elems = [
        symm_elems[i] for i, op in enumerate(initial_pg) if is_symmetric(op, graph)
    ]
    return PointGroup(final_pg_elems, ndim=graph.ndim)
