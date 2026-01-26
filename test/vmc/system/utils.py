import netket as nk
import netket_pro as nkp
import numpy as np
from netket.operator import LocalOperator


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


def Ez_ferro(Js, n_edges, h, n_nodes, S=1 / 2):
    r"""
    E_ferro = S**2* \sum_i J[i]*n_edges[i] + S*h*n_nodes
    """
    if S == 1 / 2:
        S = 1
    return S**2 * sum([J * edges for J, edges in zip(Js, n_edges)]) + S * h * n_nodes


def Exy_ferro(J, system, S=1 / 2):
    """
    Test XY ferro ground state for systems built explitictly with Sx,Sy operators
    Conceived for use with hamiltonian defined using netket_pro.operator.S_alpha operators
    """
    edges = system.graph.edges()
    hilbert = nk.hilbert.Spin(s=S, N=system.graph.n_nodes)
    xy_bonds = LocalOperator(hilbert, dtype=complex)
    for i, j in edges:
        xy_bonds += J * (
            nkp.operator.Sx(hilbert, i) * nkp.operator.Sx(hilbert, j)
            + nkp.operator.Sy(hilbert, i) * nkp.operator.Sy(hilbert, j)
        )

    e_gs = nk.exact.lanczos_ed(xy_bonds)[0]
    return e_gs


def E_exact_shastry(N, J1, J2):
    """
    Exact ground state energy for the Shastry-Sutherland model, E = N * -3*J2/2, for J2 > 2J1,
    see Shastry 1981 (remember hamiltonian defined such that S^z = +-1).
    """
    if J2 > 2 * J1 and J2 > 0 and J1 > 0:
        return -N * J2 * 3 / 2
    else:
        raise ValueError(
            "AF Shastry-Sutherland exact ground state energy only known for AF J2,J1 and J2 > 2J1"
        )


def E_exact_hyperkagome(N):
    """
    ED values per site quoted in Hutak 24 paper: -0.45374, -0.44633, -0.445100
    """
    if N == 12:
        return N * -0.45374 * 4
    elif N == 24:
        return N * -0.44633 * 4
    elif N == 36:
        return N * -0.445100 * 4
    else:
        raise ValueError("Exact ground state energy only known for N <= 36")


# TODO expected_patching_as1d functions


def expected_patching_as2d_square(nbatches: int, L: int, b: int):
    """
    Return the expected output of Square_Heisenberg.extract_patches_as2d(x, b, ...)
    for input x of shape (nbatches, L**2) with x[i,:] = [0,1,...,nsites-1]
    """
    result = np.zeros((nbatches, L // b, L // b, b**2), dtype=int)
    for i in range(L // b):
        for j in range(L // b):
            index = b * j + b * i * L
            patch = np.sort(
                np.array([index + u + L * v for u in range(b) for v in range(b)])
            )
            result[:, i, j, :] = patch
    return result


def expected_patching_as2d_shastry(nbatches: int, L: int, b: int):
    """
    Return the expected output of Square_Heisenberg.extract_patches_as2d(x, b, ...)
    for input x of shape (nbatches, L**2) with x[i,:] = [0,1,...,nsites-1]
    """
    result = np.zeros((nbatches, L // b, L // b, b**2), dtype=int)
    for i in range(L // b):
        for j in range(L // b):
            index = b**2 * j + b * i * L
            patch = np.sort(np.array([index + u for u in range(b**2)]))
            result[:, i, j, :] = patch
    return result


def expected_patching_as2d_kagome(nbatches: int, L: int):
    """
    Return the expected output of Kagome_Heisenberg.extract_patches_as2d(x, b, ...)
    for input x of shape (nbatches, 3*L**2) with x[0,:] = [0,1,...,nsites-1]
    """
    result = np.zeros((nbatches, L, L, 3), dtype=int)
    for i in range(L):
        for j in range(L):
            index = 3 * j + 3 * i * L
            patch = np.sort(np.array([index + u for u in range(3)]))
            print(patch)
            result[:, i, j, :] = patch
    return result
