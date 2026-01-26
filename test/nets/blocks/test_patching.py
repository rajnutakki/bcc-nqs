import pytest
from nets.blocks.patching import Patching
import netket as nk
import nk_extensions as nke
import numpy as np
import itertools


# Helper functions to compute expected sites in patches
def patch_sites1d(ip, patch_size):
    s0 = ip * patch_size
    sites = [s0 + k for k in range(patch_size)]
    return np.array(sites)


def patch_sites2d(ip, jp, patch_shape, lattice_shape):
    p1, p2 = patch_shape  # size of patches
    L1, L2 = lattice_shape  # size of unpatched lattice
    _, Np2 = L1 // p1, L2 // 2  # size of patched lattice

    s0 = jp * p2 + ip * p1 * p2 * Np2  # bottom left site of the patch
    sites = [s0 + k1 * L2 + k2 for k1 in range(p1) for k2 in range(p2)]
    return np.array(sites)


def patch_sites3d(ip, jp, kp, patch_shape, lattice_shape):
    p1, p2, p3 = patch_shape
    L1, L2, L3 = lattice_shape
    _, Np2, Np3 = L1 // p1, L2 // p2, L3 // p3

    s0 = (
        kp * p3 + jp * p2 * p3 * Np3 + ip * p1 * p2 * p3 * Np2 * Np3
    )  # (0,0,0) site of the of the patch
    sites = [
        s0 + k1 * L2 * L3 + k2 * L3 + k3
        for k1 in range(p1)
        for k2 in range(p2)
        for k3 in range(p3)
    ]
    return np.array(sites)


def patch_sites2d_unit(ip, jp, nb, patch_shape, lattice_shape):
    """
    Return the indices of the sites in the (ip, jp) patch of the lattice.
    Args:
        ip: index of patch in first direction
        jp: index of patch in second direction
        nb: number of sites per unit cell
        patch_shape: shape of the patch (number of unit cells in each direction)
        lattice_shape: shape of the lattice (number of unit cells in each direction)
    """
    if patch_shape is None:
        patch_shape = (1, 1)
    L1, L2 = lattice_shape  # number of unit cells in each direction
    p1, p2 = patch_shape  # number of unit cells in the patch
    _, Lp2 = L1 // p1, L2 // p2  # number of patches in each direction
    s0 = nb * (jp * p2 + ip * L2 * p1)
    sites = []
    for n in range(p1):
        for m in range(p2):
            sites += [s0 + m * nb + n * Lp2 * p2 * nb + delta for delta in range(nb)]
    return np.array(sites)


def patch_sites3d_unit(ip, jp, kp, nb, patch_shape, lattice_shape):
    """
    Return the indices of the sites in the (ip, jp, kp) patch of the lattice.
    Args:
        ip: index of patch in first direction
        jp: index of patch in second direction
        kp: index of patch in third direction
        nb: number of sites per unit cell
        patch_shape: shape of the patch (number of unit cells in each direction)
        lattice_shape: shape of the lattice (number of unit cells in each direction)
    """
    if patch_shape is None:
        patch_shape = (1, 1, 1)
    L1, L2, L3 = lattice_shape  # number of unit cells in each direction
    p1, p2, p3 = patch_shape  # number of unit cells in the patch
    _, Lp2, Lp3 = L1 // p1, L2 // p2, L3 // p3  # number of patches in each direction
    s0 = nb * (kp + jp * p2 * L3 + ip * L2 * L3 * p1)
    sites = []
    for n in range(p1):
        for m in range(p2):
            for o in range(p3):
                sites += [
                    s0
                    + o * nb
                    + m * nb * Lp3 * p3
                    + n * Lp2 * p2 * Lp3 * p3 * nb
                    + delta
                    for delta in range(nb)
                ]
    return np.array(sites)


#########


# Single-site unit cell lattices, 1D, 2D, 3D, output_dim = 1
@pytest.mark.parametrize(
    "graph, patch_shape, batch_shape, site_fn",
    [
        # 1D Chain
        (nk.graph.Chain(length=12, pbc=True), (2,), (5,), patch_sites1d),
        (nk.graph.Chain(length=12, pbc=True), (3,), (5, 6), patch_sites1d),
        # 2D Square
        (nk.graph.Square(length=12, pbc=True), (2, 2), (1,), patch_sites2d),
        (nk.graph.Square(length=12, pbc=True), (3, 2), (5, 3), patch_sites2d),
        # 2D Triangular
        (nk.graph.Triangular(extent=(6, 6), pbc=True), (2, 2), (1,), patch_sites2d),
        (nk.graph.Triangular(extent=(6, 6), pbc=True), (3, 2), (5, 3), patch_sites2d),
        # 3D Cube
        (nk.graph.Cube(length=6, pbc=True), (2, 2, 2), (5,), patch_sites3d),
        (nk.graph.Cube(length=6, pbc=True), (2, 2, 3), (5,), patch_sites3d),
        # 3D BCC
        (nk.graph.BCC(extent=(6, 4, 4), pbc=True), (2, 2, 2), (5,), patch_sites3d),
        (nk.graph.BCC(extent=(6, 4, 4), pbc=True), (2, 1, 1), (5,), patch_sites3d),
    ],
    ids=[
        # 1D
        "1D_Chain_patch(2,)_batch(5,)",
        "1D_Chain_patch(3,)_batch(5,6)",
        # 2D Square
        "2D_Square_patch(2,2)_batch(1,)",
        "2D_Square_patch(3,2)_batch(5,3)",
        # 2D Triangular
        "2D_Triangular_patch(2,2)_batch(1,)",
        "2D_Triangular_patch(3,2)_batch(5,3)",
        # 3D Cube
        "3D_Cube_patch(2,2,2)_batch(5,)",
        "3D_Cube_patch(2,2,3)_batch(5,)",
        # 3D BCC
        "3D_BCC_patch(2,2,2)_batch(5,)",
        "3D_BCC_patch(2,1,1)_batch(5,)",
    ],
)
def test_patching_output1d(graph, patch_shape, batch_shape, site_fn):
    graph = graph
    patch_size = np.prod(patch_shape)
    n_nodes = graph.n_nodes
    extent = graph.extent

    patched_extent = tuple(e // p for e, p in zip(extent, patch_shape))
    npatches = np.prod(patched_extent)

    patches = Patching(graph, output_dim=1, patch_shape=patch_shape)
    input_shape = batch_shape + (n_nodes,)
    test_input = np.full(input_shape, np.arange(n_nodes))

    test_output = patches.extract_patches(test_input)

    expected_shape = batch_shape + (npatches,) + (patch_size,)
    assert test_output.shape == expected_shape

    # Iterate over all batch indices
    for index in itertools.product(*[range(s) for s in batch_shape]):
        for patch_index in range(npatches):
            if len(patched_extent) == 1:
                i = patch_index
                expected = site_fn(i, patch_size)
            elif len(patched_extent) == 2:
                i = patch_index // patched_extent[1]
                j = patch_index % patched_extent[1]
                expected = site_fn(i, j, patch_shape, extent)
            elif len(patched_extent) == 3:
                i = patch_index // (patched_extent[1] * patched_extent[2])
                rem = patch_index % (patched_extent[1] * patched_extent[2])
                j = rem // patched_extent[2]
                k = rem % patched_extent[2]
                expected = site_fn(i, j, k, patch_shape, extent)
            else:
                raise ValueError("Unsupported patch dimension")

            assert np.all(test_output[*index, patch_index, :] == expected)


# Single-site unit cell lattices, 2D, output_dim = 2
@pytest.mark.parametrize(
    "graph, patch_shape, batch_shape, site_fn",
    [
        # 2D Square
        (nk.graph.Square(length=12, pbc=True), (2, 2), (1,), patch_sites2d),
        (nk.graph.Square(length=12, pbc=True), (3, 2), (5, 3), patch_sites2d),
        # 2D Triangular
        (nk.graph.Triangular(extent=(6, 6), pbc=True), (2, 2), (1,), patch_sites2d),
        (nk.graph.Triangular(extent=(6, 6), pbc=True), (3, 2), (5, 3), patch_sites2d),
    ],
    ids=[
        # 2D Square
        "2D_Square_patch(2,2)_batch(1,)",
        "2D_Square_patch(3,2)_batch(5,3)",
        # 2D Triangular
        "2D_Triangular_patch(2,2)_batch(1,)",
        "2D_Triangular_patch(3,2)_batch(5,3)",
    ],
)
def test_patching_output2d(graph, patch_shape, batch_shape, site_fn):
    graph = graph
    patch_size = np.prod(patch_shape)
    n_nodes = graph.n_nodes
    extent = graph.extent

    patched_extent = tuple(e // p for e, p in zip(extent, patch_shape))
    npatches = np.prod(patched_extent)

    patches = Patching(graph, output_dim=2, patch_shape=patch_shape)
    input_shape = batch_shape + (n_nodes,)
    test_input = np.full(input_shape, np.arange(n_nodes))

    test_output = patches.extract_patches(test_input)

    expected_shape = batch_shape + patched_extent + (patch_size,)
    assert test_output.shape == expected_shape

    # Iterate over all batch indices
    for index in itertools.product(*[range(s) for s in batch_shape]):
        for patch_index in range(npatches):
            i = patch_index // patched_extent[1]
            j = patch_index % patched_extent[1]
            expected = site_fn(i, j, patch_shape, extent)

            assert np.all(test_output[*index, i, j, :] == expected)


# Multi-site unit cell lattices, 2D, 3D, output_dim = 1
@pytest.mark.parametrize(
    "graph, patch_shape, batch_shape,  site_fn",
    [
        # 2D Kagome
        (nk.graph.Kagome(extent=(6, 6), pbc=True), None, (1,), patch_sites2d_unit),
        (nk.graph.Kagome(extent=(6, 6), pbc=True), (2, 2), (5, 3), patch_sites2d_unit),
        # 3D Pyrochlore
        (
            nk.graph.Pyrochlore(extent=(6, 4, 4), pbc=True),
            None,
            (5,),
            patch_sites3d_unit,
        ),
        (
            nk.graph.Pyrochlore(extent=(6, 4, 4), pbc=True),
            (2, 2, 1),
            (5, 4),
            patch_sites3d_unit,
        ),
        # 3D BCC cubic
        (
            nke.graph.BCC_cubic(extent=(6, 6, 6), pbc=True),
            (2, 2, 1),
            (5,),
            patch_sites3d_unit,
        ),
    ],
    ids=[
        # 2D Kagome
        "2D_Kagome_patchNone_batch(1,)",
        "2D_Kagome_patch(2,2)_batch(5,3)",
        # 3D Pyrochlore
        "3D_Pyrochlore_patchNone_batch(5,)",
        "3D_Pyrochlore_patch(2,2,1)_batch(5,4)",
        # 3D BCC cubic
        "3D_BCC_cubic_patch(2,2,1)_batch(5,)",
    ],
)
def test_patchingmulti_output1d(graph, patch_shape, batch_shape, site_fn):
    graph = graph
    n_nodes = graph.n_nodes
    nb = len(graph.site_offsets)
    patch_size = nb * np.prod(patch_shape) if patch_shape is not None else nb
    npatches = graph.n_nodes // patch_size
    extent = graph.extent
    patch_extent = (
        tuple(e // p for e, p in zip(extent, patch_shape))
        if patch_shape is not None
        else tuple(extent)
    )

    patches = Patching(graph, output_dim=1, patch_shape=patch_shape)
    input_shape = batch_shape + (n_nodes,)
    test_input = np.full(input_shape, np.arange(n_nodes))

    test_output = patches.extract_patches(test_input)

    expected_shape = batch_shape + (npatches,) + (patch_size,)
    assert test_output.shape == expected_shape

    # Iterate over all batch indices
    for index in itertools.product(*[range(s) for s in batch_shape]):
        for patch_index in range(npatches):
            if len(extent) == 2:
                i = patch_index // patch_extent[1]
                j = patch_index % patch_extent[1]
                expected = site_fn(i, j, nb, patch_shape, extent)
            elif len(extent) == 3:
                i = patch_index // (patch_extent[1] * patch_extent[2])
                rem = patch_index % (patch_extent[1] * patch_extent[2])
                j = rem // patch_extent[2]
                k = rem % patch_extent[2]
                expected = site_fn(i, j, k, nb, patch_shape, extent)
            else:
                raise ValueError("Unsupported patch dimension")

            assert np.all(test_output[*index, patch_index, :] == expected)


# Multi-site unit cell lattices, 2D, 3D, output_dim = 2
@pytest.mark.parametrize(
    "graph, patch_shape, batch_shape, site_fn",
    [
        # 2D Kagome
        (nk.graph.Kagome(extent=(6, 6), pbc=True), None, (1,), patch_sites2d_unit),
        (nk.graph.Kagome(extent=(6, 6), pbc=True), (2, 2), (5, 3), patch_sites2d_unit),
    ],
    ids=[
        # 2D Kagome
        "2D_Kagome_patchNone_batch(1,)",
        "2D_Kagome_patch(2,2)_batch(5,3)",
    ],
)
def test_patchingmulti_output2d(graph, patch_shape, batch_shape, site_fn):
    graph = graph
    n_nodes = graph.n_nodes
    nb = len(graph.site_offsets)
    patch_size = nb * np.prod(patch_shape) if patch_shape is not None else nb
    npatches = graph.n_nodes // patch_size
    extent = graph.extent
    patch_extent = (
        tuple(e // p for e, p in zip(extent, patch_shape))
        if patch_shape is not None
        else tuple(extent)
    )

    patches = Patching(graph, output_dim=2, patch_shape=patch_shape)
    input_shape = batch_shape + (n_nodes,)
    test_input = np.full(input_shape, np.arange(n_nodes))

    test_output = patches.extract_patches(test_input)

    expected_shape = batch_shape + patch_extent + (patch_size,)
    assert test_output.shape == expected_shape

    # Iterate over all batch indices
    for index in itertools.product(*[range(s) for s in batch_shape]):
        for patch_index in range(npatches):
            if len(extent) == 2:
                i = patch_index // patch_extent[1]
                j = patch_index % patch_extent[1]
                expected = site_fn(i, j, nb, patch_shape, extent)
            elif len(extent) == 3:
                i = patch_index // (patch_extent[1] * patch_extent[2])
                rem = patch_index % (patch_extent[1] * patch_extent[2])
                j = rem // patch_extent[2]
                k = rem % patch_extent[2]
                expected = site_fn(i, j, k, nb, patch_shape, extent)
            else:
                raise ValueError("Unsupported patch dimension")

            assert np.all(test_output[*index, i, j, :] == expected)
