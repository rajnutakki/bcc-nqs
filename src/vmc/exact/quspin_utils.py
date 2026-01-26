from quspin.basis import spin_basis_general  # Hilbert space spin basis
from vmc.system import BCCHeisenberg
import numpy as np  # generic math functions
from netket.utils.group import PermutationGroup, Identity
from netket.graph.space_group import SpaceGroup
import netket as nk


def get_basis_bcc(
    lattice_shape: tuple, k_tuple: tuple = (0, 0, 0), up_fraction: float = 0.5
):
    """
    Return an (Nstates, Nsites) array of all the quspin basis states for the hyperkagome Heisenberg model.
    Their ordering will coincide with the ordering in the exact ground state wavefunction obtained by ED.
    Basis states are returned in S = 1/2, +-1 encoding.
    The basis states are independent of J and tetragonal_distortion (which affects point group, but not translational symmetries)
    """
    Lx, Ly, Lz = lattice_shape
    kx, ky, kz = k_tuple
    up_fraction = 0.5  # Total S^z = 0
    #################
    N = 2 * Lx * Ly * Lz
    Nup = int(up_fraction * N)
    sz_sector = 0.5 * Nup - 0.5 * (N - Nup)
    system = BCCHeisenberg(
        lattice_shape=(Lx, Ly, Lz), J=(1, 1), sz_sector=sz_sector, simple_cubic=True
    )

    # Construct translations
    spacegroupbuilder = SpaceGroup(
        system.graph, nk.utils.group.trivial_point_group(ndim=system.graph.ndim)
    )
    translations = [Identity()]
    if Lx > 1:
        translations.append(spacegroupbuilder.translation_group(0)[1])  # translation +x
    if Ly > 1:
        translations.append(spacegroupbuilder.translation_group(1)[1])  # translation +y
    if Lz > 1:
        translations.append(spacegroupbuilder.translation_group(2)[1])  # translation +z

    translation_group = PermutationGroup(translations, degree=system.graph.n_nodes)

    s = np.arange(N)
    Z = -(s + 1)  # spin inversion
    z_val = 0  # spin inversion sector, 0 symmetric, 1 antisymmetric
    translation_args = {}
    if Lx > 1:
        translation_args.update({"kxblock": (translation_group[1] @ s, kx)})
    if Ly > 1:
        translation_args.update({"kyblock": (translation_group[2] @ s, ky)})
    if Lz > 1:
        translation_args.update({"kzblock": (translation_group[3] @ s, kz)})
    # setup basis
    basis_3d = spin_basis_general(
        N=system.graph.n_nodes,  # number of lattice sites
        Nup=Nup,  # sz sector
        S="1/2",
        zblock=(Z, z_val),
        **translation_args,
    )
    print(f"Size of hilbert space {basis_3d.Ns}")
    print("Computing basis states...")
    all_states = np.zeros((basis_3d.Ns, basis_3d.N), dtype=np.int8)
    for i in range(basis_3d.Ns):
        state = np.zeros((basis_3d.N), dtype=np.int8)
        state_list = list(bin(basis_3d.states[i])[2:])
        state[basis_3d.N - len(state_list) :] = np.array(state_list, dtype=np.int8)
        all_states[i] = 2 * state - 1
    print("Done")
    return all_states
