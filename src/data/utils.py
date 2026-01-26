import numpy as np
import itertools


def min_distance(r1, r2, pbc_vecs):
    """
    Compute the minimum image distance between r1 and r2 under periodic boundary conditions.

    Args:
        r1, r2: positions, of shape (ndim,)
        pbc_vec: (ndim,ndim) array, with pbc_vec[i] the ith vector winding around perodic boundaries

    Returns:
        Minimum distance between r1 and r2 considering periodic boundary conditions.
    """
    d_min = np.linalg.norm(r1 - r2)
    ndim = len(r1)
    for shift in itertools.product([-1, 0, 1], repeat=ndim):
        pbc_shift = np.array(shift) @ pbc_vecs
        d = np.linalg.norm(r1 - r2 + pbc_shift)
        if d < d_min:
            d_min = d
    return d_min
