import netket as nk
import numpy as np


def is_symmetric(symm_op, graph: nk.graph.Graph) -> bool:
    """
    Check if all edges of the graph are left invariant by the symmetry operation
    Arguments:
        symm_op: Symmetry operation
        graph: nk.graph.Graph with graph.edges(return_color=True) defined
    Returns:
        True if all edges are left invariant by the symmetry operation, False otherwise
    """
    edges_old = np.array(
        graph.edges(return_color=True)
    )  # edges_old[n,:] = [i,j,k] with (i,j) the nodes and k the color of the edge
    old_indices = np.arange(graph.n_nodes)
    new_indices = symm_op(
        old_indices
    )  # new_indices[i] = j means that the ith node is mapped to the jth node by the symmetry operation
    edges_new = np.zeros(
        (edges_old.shape[0], edges_old.shape[1] - 1)
    )  # add edges with nodes swapped to this

    for old_index in old_indices:
        new_index = new_indices[old_index]
        edges_new[edges_old[:, :2] == old_index] = (
            new_index  # replace the old index the new index
        )
        # print(f"Replacing {edges_old[:,:2][edges_old[:,:2]==old_index]} with {new_index}")

    edges_new = np.hstack(
        (edges_new, edges_old[:, 2][:, np.newaxis])
    )  # add the last column specifying edge colors which remains the same

    # Now compare edges_new and edges_old, they should contain all of the same edges
    edges_check = edges_old.copy()
    # count = 0
    for row in edges_new:
        found_index = np.where(np.all(edges_old == row, axis=1))[
            0
        ]  # the index of where the row appears in edges_old
        if len(found_index) == 0:  # if not found
            row = np.array([row[1], row[0], row[2]])  # permute i,j
            found_index = np.where(np.all(edges_old == row, axis=1))[
                0
            ]  # try to find the row again
        edges_check[found_index] = -1  # set the row to -1

    return np.all(
        edges_check == -1
    )  # if all rows are -1, then all of the original edges are contained in edges_new
