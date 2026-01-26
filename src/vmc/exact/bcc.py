# ED using quspin for the Heisenberg Hyperkagome
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_general  # Hilbert space spin basis
from vmc.system import BCCHeisenberg
import numpy as np  # generic math functions
from netket.utils.group import PermutationGroup, Identity
from netket.graph.space_group import SpaceGroup
import netket as nk
import vmc.config.args as args

if __name__ == "__main__":
    parser = args.parser
    # Command line / config arguments
    parser.add_argument(
        "--Lx",
        type=int,
    )
    parser.add_argument(
        "--Ly",
        type=int,
    )
    parser.add_argument(
        "--Lz",
        type=int,
    )
    parser.add_argument(
        "--kx",
        type=int,
    )
    parser.add_argument(
        "--ky",
        type=int,
    )
    parser.add_argument(
        "--kz",
        type=int,
    )
    parser.add_argument(
        "--up_fraction",
        type=float,
    )
    parser.add_argument("--J", type=float, action="append")
    parser.add_argument("--tetragonal_distortion", type=float, default=1.0)
    parser.add_argument("--save_base", type=str, default="")
    args = parser.parse_args()

    # Read arguments
    Lx, Ly, Lz = args.Lx, args.Ly, args.Lz
    kx, ky, kz = args.kx, args.ky, args.kz
    J = args.J
    up_fraction = args.up_fraction
    t = args.tetragonal_distortion
    print(
        f"Args: L = {Lx}{Lz}{Lz}, k = {kx}{ky}{kz}, J = {J}, up_fraction={up_fraction}, tetragonal_distortion = {t}"
    )

    #################
    N = 2 * Lx * Ly * Lz
    Nup = int(up_fraction * N)
    sz_sector = 0.5 * Nup - 0.5 * (N - Nup)
    system = BCCHeisenberg(
        lattice_shape=(Lx, Ly, Lz),
        J=J,
        sz_sector=sz_sector,
        simple_cubic=True,
        tetragonal_distortion=t,
    )
    print("System: ", system)
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
    # define Hamiltonian
    if np.any(np.array([Lx, Ly, Lz]) < 3):  # use special edges
        print("Using special edges for cluster with side length < 3")
        edges = system.custom_edges
        Js = [[J[c], i, j] for i, j, c in edges]
    else:
        Js = [
            [J[c], edge[0], edge[1]]
            for c in range(len(J))
            for edge in system.graph.edges(filter_color=c)
        ]
    static = [["xx", Js], ["yy", Js], ["zz", Js]]
    dynamic = []
    print("Computing...")
    H = hamiltonian(static, dynamic, basis=basis_3d, dtype=np.float64)
    # v = H.eigsh(k=1, which="SA", return_eigenvectors=False)
    v, w = H.eigsh(k=1, which="SA", return_eigenvectors=True)
    print(f"Ground state energy = {v[0]:.8f}")
    print(J)
    J_string = "J=" + "".join([f"{j:.3f}," for j in J])
    print(J_string)
    np.save(
        f"{args.save_base}bcc_cubic_evalue_{Lx}{Ly}{Lz}_{kx}{ky}{kz}_{up_fraction}_{z_val}_{J_string}.npy",
        v,
    )
    np.save(
        f"{args.save_base}bcc_cubic_evector_{Lx}{Ly}{Lz}_{kx}{ky}{kz}_{up_fraction}_{z_val}_{J_string}.npy",
        w,
    )
    print("Computing J1 and J2 signs")
    if len(J) == 2:
        samples = system.hilbert_space.all_states()
        j1_helper = system.sign_helperj1
        j1_signs = (-1) ** j1_helper.compute_nx(samples)
        j2_helper1 = system.sign_helperj2_1
        j2_helper2 = system.sign_helperj2_2
        j2_signs = (-1) ** (
            j2_helper1.compute_nx(samples) + j2_helper2.compute_nx(samples)
        )
        stacked_signs = np.stack((j1_signs, j2_signs))
        np.save(
            f"bcc_cubic_{Lx}{Ly}{Lz}_{kx}{ky}{kz}_{up_fraction}_{z_val}_Jsigns.npy",
            stacked_signs,
        )
    print("Done")
