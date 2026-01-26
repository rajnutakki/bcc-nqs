import numpy as np
import itertools
import matplotlib.pyplot as plt


def reciprocal_lattice_vectors(graph, from_dims=False) -> np.ndarray:
    """
    Given a graph with pbcs in all directions, compute the reciprocal lattice vectors from the formula
    B = 2pi*I@(A^T)^-1, where A is the matrix of basis vectors A[i,:] = a_i,
    B is the matrix of reciprocal lattice vectors B[i,:] = b_i, and I is the identity matrix.
    If from_dims is false, gives the usual reciprocal lattice vectors.
    If from_dims is true use the lattice dimensions as A, so that the reciprocal lattice vectors correspond to the smallest
    displacements in k-space allowed by the PBCs, such that we return \tilde{b}_i = b_i / L_i, where L_i is the extent of the graph along direction a_i.
    Args:
        graph: a netket graph object with PBCs in all directions
        from_dims: whether to compute from lattice dimensions or basis vectors
    Return: (ndim, ndim) array, with [i,:] the i^th reciprocal lattice vector
    """
    assert np.all(graph.pbc), "Graph must have PBCs in all directions"
    if from_dims:
        A = graph._lattice_dims
    else:
        A = graph.basis_vectors
    # if graph.ndim == 2:
    #     b1 = 2*np.pi * np.array([A[1,1], -A[1,0]]) / (A[0,:].dot(np.array([A[1,1], -A[1,0]])))
    #     b2 = 2*np.pi * np.array([-A[0,1], A[0,0]]) / (A[0,:].dot(np.array([A[1,1], -A[1,0]])))
    #     B = np.array([b1,b2])

    # else:
    #     b1 = 2*np.pi * np.cross(A[1,:], A[2,:]) / (A[0,:].dot(np.cross(A[1,:], A[2,:])))
    #     b2 = 2*np.pi * np.cross(A[2,:], A[0,:]) / (A[1,:].dot(np.cross(A[2,:], A[0,:])))
    #     b3 = 2*np.pi * np.cross(A[0,:], A[1,:]) / (A[2,:].dot(np.cross(A[0,:], A[1,:])))
    #     B = np.array([b1,b2,b3])

    B = 2 * np.pi * np.eye(graph.ndim) @ np.linalg.inv(A.T)

    for i, j in itertools.product(range(graph.ndim), range(graph.ndim)):
        assert np.isclose(A[i, :].dot(B[j, :]), 2 * np.pi * (i == j)), (
            "Does not satisfy a_i . b_j = 2*pi delta_ij"
        )
    return B


def valid_wavevectors(graph):
    """
    Given a graph with PBCs, compute the valid wavevectors in Cartesian coordinates.
    Return: (Nunitcells, graph.ndim) array, with [i,:] the i^th wavevector
    """
    assert np.all(graph.pbc), "Graph must have PBCs in all directions"
    # Solve for the valid basis vectors in reciprocal space
    B = reciprocal_lattice_vectors(graph, from_dims=True)
    ranges = [
        range(-graph.extent[d] // 2, graph.extent[d] // 2) for d in range(graph.ndim)
    ]
    wavevectors = []
    for ns in itertools.product(*ranges):
        k = sum(ns[d] * B[d] for d in range(graph.ndim))
        wavevectors.append(k)

    wavevectors = np.array(wavevectors)
    assert len(wavevectors) == np.prod(
        graph.extent
    )  # Number of wavevectors should be equivalent to number of unit cells
    graph.to_reciprocal_lattice(
        wavevectors
    )  # Check they are valid on graph, note this is checking on the actual graph
    return wavevectors


class ReciprocalLattice:
    def __init__(self, graph):
        self.graph = graph
        self.basis_vectors = reciprocal_lattice_vectors(graph, from_dims=False)
        self.kpoints = valid_wavevectors(graph)

    def irreducible_points(self, pg_matrices, tol=1e-10):
        """
        Compute the kpoints in the irreducible Brillouin zone, given the previously computed set of all valid kpoints in the BZ.
        Applies the point group symmetries to each kpoint, removing all kpoints which are generated from a representative
        by a point group operation.
        Args:
            pg_matrices: (Npg, ndim, ndim) array of point group operations in matrix form
            tol: tolerance for determining if two kpoints are equivalent
        Returns:
            (Nibz_points, ndim) array of k points within the irreducible Brillouin zone
        """
        kpoints = self.kpoints
        self.pg_matrices = pg_matrices
        used = np.zeros(len(kpoints), dtype=bool)
        ibz_points = []

        for i, k in enumerate(kpoints):
            if used[i]:  # already have looked at this kpoint's associated star
                continue

            # Generate orbit of k under point group symmetries
            star = np.round(
                np.tensordot(pg_matrices, k, axes=1), decimals=int(np.log10(10 / tol))
            )  # need to round to avoid random sort errors later
            # Match against existing k-points and mark them as used
            for j, kp in enumerate(kpoints):
                if not used[j] and np.any(np.linalg.norm(star - kp, axis=1) < tol):
                    used[j] = True

            # Pick representative, sorts by kx, then ky, then kz
            representative = star[np.lexsort(star.T)][0]
            ibz_points.append(representative)

        ibz_points = np.unique(np.array(ibz_points), axis=0)
        self.ibz_points = ibz_points
        return ibz_points

    def get_representative(self, k, tol=1e-10):
        """
        Find the representative of a given k in the irreducible Brillouin zone
        """
        assert hasattr(self, "pg_matrices"), (
            "Must first compute irreducible points with point group matrices"
        )
        star = np.tensordot(self.pg_matrices, k, axes=1)
        for kp in star:
            for kipz in self.ibz_points:
                if np.linalg.norm(kp - kipz) < tol:
                    return kipz

        raise ValueError("Could not find representative in irreducible Brillouin zone")

    def plot(self, basis_vectors=True, ibz=True, **kwargs):
        """
        Scatter plot of the kpoints in the BZ
        """
        fig = plt.figure()
        if self.graph.ndim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(self.kpoints[:, 0], self.kpoints[:, 1], **kwargs)
            if basis_vectors:
                for i in range(self.graph.ndim):
                    ax.arrow(
                        self.kpoints[0, 0],
                        self.kpoints[0, 1],
                        self.basis_vectors[i, 0],
                        self.basis_vectors[i, 1],
                        color="r",
                        head_width=0.1,
                        length_includes_head=True,
                    )
            if ibz and hasattr(self, "ibz_points"):
                ax.scatter(
                    self.ibz_points[:, 0],
                    self.ibz_points[:, 1],
                    color="g",
                    s=100,
                    marker="x",
                )
            ax.set_xlabel(r"$k_x$")
            ax.set_ylabel(r"$k_y$")
            ax.set_aspect("equal")
        elif self.graph.ndim == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                self.kpoints[:, 0], self.kpoints[:, 1], self.kpoints[:, 2], **kwargs
            )
            if basis_vectors:
                for i in range(self.graph.ndim):
                    ax.quiver(
                        self.kpoints[0, 0],
                        self.kpoints[0, 1],
                        self.kpoints[0, 2],
                        self.basis_vectors[i, 0],
                        self.basis_vectors[i, 1],
                        self.basis_vectors[i, 2],
                        color="r",
                        arrow_length_ratio=0.1,
                    )
            if ibz and hasattr(self, "ibz_points"):
                ax.scatter(
                    self.ibz_points[:, 0],
                    self.ibz_points[:, 1],
                    self.ibz_points[:, 2],
                    color="g",
                    s=100,
                    marker="x",
                )
            ax.set_xlabel(r"$k_x$")
            ax.set_ylabel(r"$k_y$")
            ax.set_zlabel(r"$k_z$")
            ax.set_box_aspect([1, 1, 1])
        else:
            raise NotImplementedError("Plotting only implemented for 2D and 3D graphs")
        return ax
