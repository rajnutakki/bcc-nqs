import netket as nk
from vmc.expectation_value.observables import compute_phases
from vmc.system import BCCHeisenberg
import numpy as np


def test_phase_computation():
    system = BCCHeisenberg(lattice_shape=(2, 2, 2), J=(1, 0))

    v, w = nk.exact.lanczos_ed(system.hamiltonian, compute_eigenvectors=True)
    samples = system.hilbert_space.all_states()
    reshaped_samples = samples.reshape((10, samples.shape[0] // 10, samples.shape[1]))
    phases, j1_signs, j2_signs = compute_phases(
        reshaped_samples, np.log(w[:, 0], dtype=complex), system
    )
    signs = np.sign(np.exp(1j * phases) / np.exp(1j * phases[0]))
    if np.isclose(signs[0], j1_signs[0]):
        assert np.allclose(signs, j1_signs), "J1 signs do not match phases"
    elif np.isclose(-signs[0], j1_signs[0]):
        assert np.allclose(-signs, j1_signs), "J1 signs do not match phases"
    else:
        raise ValueError("J1 signs do not match phases")
