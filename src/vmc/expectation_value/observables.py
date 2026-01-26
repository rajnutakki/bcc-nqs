from typing import Sequence
from nk_extensions.operator.spin import Sx, Sy, Sz, Sp, Sm, Correlators
import numpy as np


class ObservableParser:
    def __init__(self, obs_str: Sequence[str], system):
        self.operators = {}
        self.compute_signs = False

        if "energy" in obs_str:
            self.operators["energy"] = system.hamiltonian

        if "mz" in obs_str:
            self.operators["mz"] = (1 / system.graph.n_nodes) * sum(
                [Sz(system.hilbert_space, i) for i in range(system.graph.n_nodes)]
            )

        if "mx" in obs_str:
            self.operators["mx"] = (1 / system.graph.n_nodes) * sum(
                [Sx(system.hilbert_space, i) for i in range(system.graph.n_nodes)]
            )

        if "sx" in obs_str:
            self.operators["sx"] = sum(
                [Sx(system.hilbert_space, i) for i in range(system.graph.n_nodes)]
            )

        if "sy" in obs_str:
            self.operators["sy"] = sum(
                [Sy(system.hilbert_space, i) for i in range(system.graph.n_nodes)]
            )

        if "mp" in obs_str:
            self.operators["mp"] = (1 / system.graph.n_nodes) * sum(
                [Sp(system.hilbert_space, i) for i in range(system.graph.n_nodes)]
            )

        if "mm" in obs_str:
            self.operators["mm"] = (1 / system.graph.n_nodes) * sum(
                [Sm(system.hilbert_space, i) for i in range(system.graph.n_nodes)]
            )

        if "sz_sq" in obs_str:
            self.operators["sz_sq"] = (1 / system.graph.n_nodes) * sum(
                [
                    Sz(system.hilbert_space, i) * Sz(system.hilbert_space, i)
                    for i in range(system.graph.n_nodes)
                ]
            )

        if "szsz_t" in obs_str:
            corr = Correlators(
                Sz,
                Sz,
                system.hilbert_space,
                system.graph,
                transl_invariant=True,
                name="SzSz_t",
            )
            self.operators.update(corr.local_operators)

        if "szsz" in obs_str:
            corr = Correlators(
                Sz,
                Sz,
                system.hilbert_space,
                system.graph,
                transl_invariant=False,
                name="SzSz",
            )
            self.operators.update(corr.local_operators)

        if "spsm_t" in obs_str:
            corr = Correlators(
                Sp,
                Sm,
                system.hilbert_space,
                system.graph,
                transl_invariant=True,
                name="SpSm_t",
            )
            self.operators.update(corr.local_operators)

        if "spsm" in obs_str:
            corr = Correlators(
                Sp,
                Sm,
                system.hilbert_space,
                system.graph,
                transl_invariant=False,
                name="SpSm",
            )
            self.operators.update(corr.local_operators)

        if "sign" in obs_str:
            print("Compute signs = True")
            self.compute_signs = True


def compute_phases(samples, logvalues, system):
    """
    Given an array of samples, compute the phases of the wavefunction, \phi(\sigma)
    of the wavefunction where \Psi(\sigma) = |\Psi(\sigma)| exp(i \phi(\sigma)).
    Also computes the sign(\sigma) given by Marshall sign rules of the system.
    At the moment just designed for the bcc system, but could be easily extended to others.
    Input: samples: array of shape (..., Ns) where Ns is number of spins
           logvalues: log\Psi(samples)
           system: system object containing objects to compute sign rules

    Returns: phases: flattened array of \phi(\sigma)
             j1_signs: flattened array of Marshall sign from J1 only Hamiltonian
             j2_signs: flattened array of Marshall sign from J2 only Hamiltonian
    """
    print("Computing sign information...")
    phases = np.angle(np.exp(logvalues))
    j1_helper = system.sign_helperj1
    j1_signs = (-1) ** j1_helper.compute_nx(samples)
    j2_helper1 = system.sign_helperj2_1
    j2_helper2 = system.sign_helperj2_2
    j2_signs = (-1) ** (j2_helper1.compute_nx(samples) + j2_helper2.compute_nx(samples))
    return phases.flatten(), j1_signs.flatten(), j2_signs.flatten()
