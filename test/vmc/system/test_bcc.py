import pytest
import jax
from vmc.system import BCCHeisenberg
from nets.net import ViTNd
from nk_extensions.group.translations import translation_group_from_axis_translations
import numpy as np
from netket.utils.group import cubic
from nk_extensions.group.pg_utils import extract_valid_point_group
import nk_extensions as nke
import netket as nk

test_input = (
    pytest.param(
        BCCHeisenberg(lattice_shape=(4, 4, 4), J=(1, 2)),
        48,
        id="BCCHeisenberg(lattice_shape=(4,4,4), J=(1,2))",
    ),
    pytest.param(
        BCCHeisenberg(lattice_shape=(4, 4, 4), J=(1, 2), tetragonal_distortion=0.985),
        16,
        id="BCCHeisenberg(lattice_shape=(4,4,4), J=(1,2), tetragonal_distortion=0.985)",
    ),
    pytest.param(
        BCCHeisenberg(lattice_shape=(4, 4, 6), J=(1, 2)),
        16,
        id="BCCHeisenberg(lattice_shape=(6,4,4), J=(1,2))",
    ),
    pytest.param(
        BCCHeisenberg(lattice_shape=(4, 4, 6), J=(1, 2), tetragonal_distortion=0.985),
        16,
        id="BCCHeisenberg(lattice_shape=(6,4,4), J=(1,2), tetragonal_distortion=0.985)",
    ),
    pytest.param(
        BCCHeisenberg(lattice_shape=(6, 5, 4), J=(1, 2)),
        8,
        id="BCCHeisenberg(lattice_shape=(6,5,4), J=(1,2)",
    ),
    pytest.param(
        BCCHeisenberg(lattice_shape=(6, 5, 4), J=(1, 2), tetragonal_distortion=0.985),
        8,
        id="BCCHeisenberg(lattice_shape=(6,5,4), J=(1,2), tetragonal_distortion=0.985",
    ),
)


@pytest.mark.parametrize("system, expected_pglengths", test_input)
def test_bcc_symmetries(system, expected_pglengths):
    # Check the point group has the right number of elements
    assert len(system.point_group) == expected_pglengths
    # Check that the permutation group can act on a random state
    state = 2 * np.random.randint(low=0, high=2, size=system.graph.n_nodes) - 1
    for elem in system.point_group:
        _ = elem @ state


@pytest.mark.parametrize(
    "lattice_shape, patch_shape, tetragonal_distortion",
    [
        pytest.param((4, 2, 2), (1, 1, 1), 1, id="BCCcubic_(422)_patch(111)"),
        pytest.param((4, 2, 2), (2, 1, 1), 1, id="BCCcubic_(422)_patch(211)"),
        pytest.param((4, 4, 4), (1, 1, 1), 1, id="BCCcubic_(444)_patch(111)"),
        pytest.param((4, 4, 4), (2, 2, 1), 1, id="BCCcubic_(444)_patch(221)"),
        pytest.param((6, 6, 6), (1, 1, 1), 1, id="BCCcubic_(666)_patch(111)"),
        pytest.param((6, 6, 6), (2, 2, 1), 1, id="BCCcubic_(666)_patch(221)"),
        pytest.param((4, 2, 2), (1, 1, 1), 0.985, id="BCCcubic_(422)_patch(111)"),
        pytest.param((4, 2, 2), (2, 1, 1), 0.985, id="BCCcubic_(422)_patch(211)"),
        pytest.param((4, 4, 4), (1, 1, 1), 0.985, id="BCCcubic_(444)_patch(111)"),
        pytest.param((4, 4, 4), (2, 2, 1), 0.985, id="BCCcubic_(444)_patch(221)"),
        pytest.param((6, 6, 6), (1, 1, 1), 0.985, id="BCCcubic_(666)_patch(111)"),
        pytest.param((6, 6, 6), (2, 2, 1), 0.985, id="BCCcubic_(666)_patch(221)"),
    ],
)
def test_vit_bcc_invariance(lattice_shape, patch_shape, tetragonal_distortion):
    system = BCCHeisenberg(
        lattice_shape=lattice_shape,
        J=(1, 2),
        patch_shape=patch_shape,
        tetragonal_distortion=tetragonal_distortion,
    )
    net = ViTNd(
        depth=2,
        d_model=12,
        heads=6,
        output_head="Vanilla",
        expansion_factor=2,
        system=system,
        q=(0, 0, 0),
        kernel_shape=(2, 2, 2),
        patch_shape=patch_shape,
    )
    Tgroup = translation_group_from_axis_translations(
        system.graph, n=patch_shape
    )  # interpatch translations
    sample = system.hilbert_space.random_state(key=jax.random.PRNGKey(0))
    params = net.network.init(jax.random.PRNGKey(0), sample)
    translated_samples = Tgroup @ sample
    translated_samples2 = system.translation_group @ sample
    outputs = net.network.apply(params, translated_samples)
    outputs2 = net.network.apply(params, translated_samples2)
    assert not np.allclose(outputs2, outputs2[0]), (
        "Should not be invariant under full translation group"
    )
    assert np.allclose(outputs, outputs[0]), (
        "Should be invariant under interpatch translation group"
    )


# Test invariant under all stages of symmetrization
@pytest.mark.parametrize(
    "lattice_shape, patch_shape, tetragonal_distortion",
    [
        pytest.param((4, 2, 2), (1, 1, 1), 1, id="BCCcubic_(422)_patch(111)"),
        pytest.param((4, 2, 2), (2, 1, 1), 1, id="BCCcubic_(422)_patch(211)"),
        pytest.param((4, 2, 2), (1, 1, 1), 0.985, id="BCCcubic_(422)_patch(111)"),
        pytest.param((4, 2, 2), (2, 1, 1), 0.985, id="BCCcubic_(422)_patch(211)"),
    ],
)
def test_bcc_symmetrization(lattice_shape, patch_shape, tetragonal_distortion):
    system = BCCHeisenberg(
        lattice_shape=lattice_shape,
        J=(1, 2),
        patch_shape=patch_shape,
        tetragonal_distortion=tetragonal_distortion,
    )
    net = ViTNd(
        depth=2,
        d_model=12,
        heads=6,
        output_head="Vanilla",
        expansion_factor=2,
        system=system,
        q=(0, 0, 0),
        kernel_shape=(2, 2, 2),
        patch_shape=patch_shape,
    )
    symmetry_ops = [
        translation_group_from_axis_translations(
            system.graph, n=patch_shape
        ),  # translations up to patches
        system.translation_group,  # all translations
        system.translation_group @ system.point_group,  # full space group
    ]
    nets = [f(net.network) for f in system.symmetrizing_functions[:-1]]
    for i, (symm_op, net) in enumerate(zip(symmetry_ops, nets)):
        print(i)
        sample = system.hilbert_space.random_state(key=jax.random.PRNGKey(0))
        params = net.init(jax.random.PRNGKey(0), sample)
        symm_samples = symm_op @ sample
        outputs = net.apply(params, symm_samples)
        assert np.allclose(outputs, outputs[0]), (
            f"Network not invariant under {symm_op} symmetrization"
        )


# Test correct transformations according to little group
@pytest.mark.parametrize(
    "lattice_shape, tetragonal_distortion, patch_shape",
    [
        pytest.param((4, 2, 2), 1, None, id="BCC_(422_cubic)_t1"),
        pytest.param((4, 2, 2), 0.985, None, id="BCC_(422_cubic)_t0.985"),
        pytest.param((4, 2, 2), 1, (2, 1, 1), id="BCC_(422_cubic)_t1_patched"),
        pytest.param((4, 2, 2), 0.985, (2, 1, 1), id="BCC_(422_cubic)_t0.985_patched"),
    ],
)
def test_bcc_littlegroup_symmetrization(
    lattice_shape, tetragonal_distortion, patch_shape
):
    J = (1, 2)
    temp_system = BCCHeisenberg(
        lattice_shape=lattice_shape,
        J=J,
        tetragonal_distortion=tetragonal_distortion,
        patch_shape=patch_shape,
    )
    ks = nke.graph.reciprocal_space.valid_wavevectors(temp_system.graph)
    ks = [
        k for k in ks if k[1] >= k[2]
    ]  # Some momenta equivalent by symmetry, remove these
    for k in ks:
        k = k / np.pi
        print(" k = ", k)
        temp_system = BCCHeisenberg(
            lattice_shape=lattice_shape,
            J=J,
            tetragonal_distortion=tetragonal_distortion,
            q=k,
            patch_shape=patch_shape,
        )
        # point_group = extract_valid_point_group(temp_system.graph, cubic.Oh())
        space_group = temp_system.graph.space_group()
        little_group = space_group.little_group(k * np.pi)
        print(little_group)
        for i_id in range(len(little_group.character_table())):
            print("Irrep id = ", i_id)
            if np.isclose(
                np.sum(abs(little_group.character_table()[i_id])),
                little_group.character_table().shape[1],
            ):
                print("1D irrep, can check correct transformation properties")
                system = BCCHeisenberg(
                    lattice_shape=lattice_shape,
                    J=J,
                    tetragonal_distortion=tetragonal_distortion,
                    q=k,
                    little_group_id=i_id,
                    patch_shape=patch_shape,
                )
                net = ViTNd(
                    depth=2,
                    d_model=12,
                    heads=6,
                    output_head="FT",
                    expansion_factor=2,
                    system=system,
                    q=k,
                    kernel_shape=(2, 2, 2),
                    patch_shape=patch_shape,
                )
                symm_op = system.graph.point_group(little_group)  # little group
                symm_net = system.symmetrizing_functions[1](net.network)
                sample = system.hilbert_space.random_state(
                    key=jax.random.PRNGKey(0), size=1
                )
                params = symm_net.init(jax.random.PRNGKey(0), sample)
                symm_samples = symm_op @ sample[0]
                outputs = np.exp(symm_net.apply(params, symm_samples))
                assert np.allclose(
                    outputs / outputs[0], little_group.character_table()[i_id]
                ), f"Network doesnt transform as expected for little_group_id {i_id}"


# Test correct transformations according to momentum
@pytest.mark.parametrize(
    "lattice_shape, tetragonal_distortion, patch_shape",
    [
        pytest.param((4, 2, 2), 1, None, id="BCC_(422_cubic)_t1"),
        pytest.param((4, 2, 2), 0.985, None, id="BCC_(422_cubic)_t0.985"),
    ],
)
def test_bcc_spacegroup_symmetrization(
    lattice_shape, tetragonal_distortion, patch_shape
):
    J = (1, 2)
    temp_system = BCCHeisenberg(
        lattice_shape=lattice_shape,
        J=J,
        tetragonal_distortion=tetragonal_distortion,
        patch_shape=patch_shape,
    )
    ks = nke.graph.reciprocal_space.valid_wavevectors(temp_system.graph)
    ks = [
        k for k in ks if k[1] >= k[2]
    ]  # Some momenta equivalent by symmetry, remove these
    translation_group = temp_system.graph.translation_group()
    rts = np.array(
        [np.array([0, 0, 0])]
        + [
            t._vector.dot(temp_system.graph.basis_vectors)
            for t in translation_group.elems[1:]
        ]
    )
    for k in ks:
        k = k / np.pi
        print(" k = ", k)
        temp_system = BCCHeisenberg(
            lattice_shape=lattice_shape,
            J=J,
            tetragonal_distortion=tetragonal_distortion,
            q=k,
            patch_shape=patch_shape,
        )
        point_group = extract_valid_point_group(temp_system.graph, cubic.Oh())
        space_group = temp_system.graph.space_group(point_group)
        little_group = space_group.little_group(k * np.pi)
        for i_id in range(len(little_group.character_table())):
            print("Irrep id = ", i_id)
            system = BCCHeisenberg(
                lattice_shape=lattice_shape,
                J=J,
                tetragonal_distortion=tetragonal_distortion,
                q=k,
                little_group_id=i_id,
                patch_shape=patch_shape,
            )
            net = ViTNd(
                depth=2,
                d_model=12,
                heads=6,
                output_head="FT",
                expansion_factor=2,
                system=system,
                q=k,
                kernel_shape=(2, 2, 2),
                patch_shape=patch_shape,
            )
            symm_op = translation_group
            symm_net = system.symmetrizing_functions[2](net.network)
            sample = system.hilbert_space.random_state(
                key=jax.random.PRNGKey(0), size=1
            )
            params = symm_net.init(jax.random.PRNGKey(0), sample)
            symm_samples = symm_op @ sample[0]
            outputs = np.exp(symm_net.apply(params, symm_samples))
            expected_character = np.exp(-1j * np.pi * rts @ k)
            print(outputs / outputs[0], expected_character)
            assert np.allclose(outputs / outputs[0], expected_character), (
                f"Network doesnt transform as expected for little_group_id {i_id}, k = {k}"
            )
            print(f"Passed for {i_id}")


# Test energies of ferromagnetic Hamiltonians to check correct number of edges for small clusters
def E_ferro(N, J):
    if len(J) == 2:  # without tetragonal distortion
        return N * (4 * J[0] + 3 * J[1])
    if len(J) == 3:  # with tetragonal distortion
        return N * (4 * J[0] + J[1] + 2 * J[2])


@pytest.mark.parametrize(
    "system",
    [
        pytest.param(
            BCCHeisenberg(lattice_shape=(2, 2, 2), J=(-2, -1)),
            id="BCC_(222)_J(-2,-1)_nodistortion",
        ),
        pytest.param(
            BCCHeisenberg(
                lattice_shape=(2, 2, 2), J=(-2, -1, -0.5), tetragonal_distortion=0.985
            ),
            id="BCCcubic_(222)_J(-2,-1,-0,5)_distortion",
        ),
    ],
)
def test_bcc_energies(system):
    E0 = nk.exact.lanczos_ed(system.hamiltonian, k=1)[0]
    E_analytical = E_ferro(system.graph.n_nodes, system.J)
    assert np.isclose(E0, E_analytical), (
        f"Energy {E0} does not match analytical {E_analytical}"
    )
