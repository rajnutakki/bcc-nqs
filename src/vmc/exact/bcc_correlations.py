# Compute the SzSz correlations of an eigenvector of the BCC Heisenberg model

import numpy as np  # generic math functions
import vmc.config.args as args
from vmc.exact.quspin_utils import get_basis_bcc

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
    parser.add_argument(
        "--eigenvector_file",
        type=str,
    )
    args = parser.parse_args()

    # Read arguments
    lattice_shape = (args.Lx, args.Ly, args.Lz)
    k_tuple = (args.kx, args.ky, args.kz)
    ni = get_basis_bcc(
        lattice_shape=lattice_shape, k_tuple=k_tuple, up_fraction=args.up_fraction
    )  # (Nstates, Nsites) array of \sigma_i^z values
    print(f"ni shape: {ni.shape}")
    cn = np.load(args.eigenvector_file)[:, 0]
    assert ni.shape[0] == cn.shape[0], (
        f"Number of basis states {ni.shape[0]} does not match number of eigenvector entries {cn.shape[0]}"
    )
    cn = cn.reshape(-1, 1)  # reshape cn to (Nstates, 1) to match ni
    ci = np.conj(cn) * ni
    cj = cn * ni
    print(ci.shape, cj.shape)
    corr = ci.T @ cj  # (Nsites, Nsites) array of <\sigma_i^z \sigma_j^z>
    save_file = args.eigenvector_file[:-4] + "_szsz.npy"
    np.save(save_file, corr)
    print(f"Saved SzSz correlations to {save_file}")
