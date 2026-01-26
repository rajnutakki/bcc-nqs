# For computing static spin structure factor and plotting real-space spin correlations
import numpy as np
from netket.graph import Graph
from data.utils import min_distance
from typing import Optional


class SpinCorr:
    def __init__(
        self, S: np.ndarray, graph: Graph, S_error: Optional[np.ndarray] = None
    ):
        """
        Create SpinCorr class with matrix of spin-spin correlations.
        Input:
            S: (Nbasis, N) or (N,N) array with elements S[i,j] = S^{\alpha}_i S^{\beta}_j, where N is
            total number of lattice sites and Nbasis is the number of sites in the unit cell.
            graph: graph on which the spin correlations were computed
            S_error[optional]: (Nbasis, N) or (N,N) array with statistical error on S[i,j]
        """
        assert S.ndim == 2
        if S.shape[0] == S.shape[1]:
            print("N_i = N_j, assuming S is full (N,N) correlation matrix")
            self.full = True

        else:
            assert S.shape[0] == len(graph.site_offsets), (
                "First dimension should match number of sites in unit cell"
            )
            print("N_i != N_j, assuming S is (Nbasis, N) correlation matrix")
            self.full = False

        if S_error is not None:
            assert S_error.shape == S.shape, "S_error should have the same shape as S"

        self.S = S
        self.S_error = S_error
        self.Ni = S.shape[0]
        self.Nj = S.shape[1]
        self.graph = graph
        # Compute R[i,j] = r_i - r_j
        self.R = np.zeros((self.Ni, self.Nj, self.graph.ndim))
        for i in range(self.Ni):
            for j in range(self.Nj):
                self.R[i, j] = self.graph.positions[i] - self.graph.positions[j]

        for i in range(self.graph.ndim):
            assert np.allclose(np.diag(self.R[:, :, i]), 0)

    def distances(self):
        """
        Returns: R_min = (N,N) array of minimum distance between sites i and j (taking into account pbcs)
                 corresponding to self.S[i,j]

                 Can plot correlations as a function of distance with e.g plt.plot(R_min.flatten(), S.flatten(), 'o')
        """
        # Smallest non-zero vector which maps lattice back to itself with pbcs
        pbc_vec = np.array(
            [
                (self.graph._pbc[i] * self.graph._lattice_dims[i])[i]
                for i in range(self.graph.ndim)
            ]
        )
        R_min = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                R_min[i, j] = min_distance(
                    self.graph.positions[i], self.graph.positions[j], pbc_vec
                )

        return R_min

    def structure_factor(self, Q):
        """
        Compute S(Q) = 1/N * sum_{i,j} S_{ij} exp(i Q.R_{ij})
        where R_{ij} = r_i - r_j is the vector between sites i and j
        and Q is a tensor of Q values.
        Input: Q (..., ndim) array of q values
        Output: S_Q (...,) array of structure factor computed for each q contained in Q
                S_Q_err (...,) array of statistical error on S_Q if S_error was provided, else None
        """
        S_q_err = None
        if self.full:
            norm = self.Nj
        else:
            norm = self.Ni
        assert Q.shape[-1] == self.graph.ndim, "Q must have the same dimension as graph"
        _ = self.graph.to_reciprocal_lattice(
            ks=Q
        )  # Check if the Q values correspond to the reciprocal lattice, raises error if not
        exp_matrix = np.exp(1j * np.einsum("...d,ijd->...ij", Q, self.R))  # e^{iqr}
        S_q = np.einsum("ij,...ij->...", self.S, exp_matrix) / norm
        if self.S_error is not None:
            S_q_err = (
                np.sqrt(
                    np.einsum("ij,...ij->...", self.S_error**2, np.abs(exp_matrix) ** 2)
                )
                / norm
            )

        return S_q, S_q_err
