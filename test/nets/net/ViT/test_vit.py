import pytest
from nets.net import ViTNd
from nets.blocks.patching import Patching
from vmc.system import BCCHeisenberg
import jax
import jax.numpy as jnp


@pytest.mark.parametrize("system", (BCCHeisenberg(lattice_shape=(4, 2, 2), J=(1, 2)),))
def test_PositiveHead(system):
    net = ViTNd(
        depth=2,
        d_model=12,
        heads=6,
        output_head="Positive",
        expansion_factor=2,
        q=(0, 0, 0),  # dummy,
        sign_net=False,
        system=system,
    )

    vars = net.network.init(
        jax.random.PRNGKey(0),
        system.hilbert_space.random_state(jax.random.PRNGKey(0), size=1),
    )
    samples = system.hilbert_space.random_state(jax.random.PRNGKey(0), size=20)
    print(samples.shape)
    logpsis = net.network.apply(vars, samples)
    psis = jnp.exp(logpsis)
    print(psis)
    assert jnp.all(psis > 0)


@pytest.mark.parametrize(
    "Ls", [(i, j, k) for i in range(2, 4) for j in range(2, 4) for k in range(2, 4)]
)
def test_FTHeadViTBCC(Ls):
    # Check correct tranformations under translations
    lattice_shape = Ls
    N = 2 * jnp.prod(jnp.array(Ls))
    system = BCCHeisenberg(lattice_shape=lattice_shape, J=(1,))
    depth = 2
    d_model = 12
    heads = 6
    output_head_name = "FT"
    expansion_factor = 2
    patches = Patching(system.graph, output_dim=1)
    # For all momenta commensurate with lattice
    for qx in jnp.linspace(0, 2, Ls[0], endpoint=False):
        for qy in jnp.linspace(0, 2, Ls[1], endpoint=False):
            for qz in jnp.linspace(0, 2, Ls[2], endpoint=False):
                q = (qx, qy, qz)
                print(f"q = {q}")
                net = ViTNd(
                    depth=depth,
                    d_model=d_model,
                    heads=heads,
                    output_head=output_head_name,
                    expansion_factor=expansion_factor,
                    q=q,
                    sign_net=False,
                    system=system,
                )

                sample = jnp.ones((5, N))
                vars = net.network.init(jax.random.PRNGKey(0), sample)
                # Compute output on random sample translated by all elements of symmetry group
                translation_group = system.graph.translation_group()
                sample = system.hilbert_space.random_state(jax.random.PRNGKey(0))
                translated_samples = translation_group @ sample
                logpsi_q = net.network.apply(vars, translated_samples)
                psi_q = jnp.exp(logpsi_q)
                abs_psi = jnp.abs(psi_q)
                arg_psi = jnp.angle(psi_q)
                positions = patches.compute_positions()
                phases = positions @ jnp.array(q) % (2 * jnp.pi)
                phases = jnp.floor(phases / (2 * jnp.pi)) * 2 * jnp.pi  # round down,
                # to avoid cases where phase is just under 2pi due to being float
                print("|z| = ", abs_psi)
                print("arg(z) = ", arg_psi)
                print("q.r % 2pi = ", phases)
                # Check properties of output are correct
                # 1. Absolute values are all the same
                assert jnp.allclose(abs_psi, abs_psi[0])
                unique_args = jnp.unique(jnp.round(arg_psi, decimals=10))
                # 2. The phase corresponds to q*r
                for unique_arg in unique_args:
                    arg_indices = jnp.isclose(unique_arg, arg_psi)
                    print(
                        "isclose",
                        jnp.isclose(phases[arg_indices][0], phases[arg_indices]),
                    )
                    print(
                        phases[arg_indices][
                            jnp.invert(
                                jnp.isclose(phases[arg_indices][0], phases[arg_indices])
                            )
                        ]
                    )
                    assert jnp.allclose(phases[arg_indices][0], phases[arg_indices])


@pytest.mark.parametrize(
    "Ls", [(i, j, k) for i in range(2, 4) for j in range(2, 4) for k in range(2, 4)]
)
def test_FTHeadViTBCCSign(Ls):
    # Check correct tranformations under translations
    lattice_shape = Ls
    N = 2 * jnp.prod(jnp.array(Ls))
    system = BCCHeisenberg(lattice_shape=lattice_shape, J=(1,), sign_rule=1)
    depth = 2
    d_model = 12
    heads = 6
    output_head_name = "FT"
    expansion_factor = 2
    patches = Patching(system.graph, output_dim=1)
    # For all momenta commensurate with lattice
    for qx in jnp.linspace(0, 2, Ls[0], endpoint=False):
        for qy in jnp.linspace(0, 2, Ls[1], endpoint=False):
            for qz in jnp.linspace(0, 2, Ls[2], endpoint=False):
                q = (qx, qy, qz)
                print(f"q = {q}")
                net = ViTNd(
                    depth=depth,
                    d_model=d_model,
                    heads=heads,
                    output_head=output_head_name,
                    expansion_factor=expansion_factor,
                    q=q,
                    sign_net=True,
                    system=system,
                )

                sample = jnp.ones((5, N))
                vars = net.network.init(jax.random.PRNGKey(0), sample)
                # Compute output on random sample translated by all elements of symmetry group
                translation_group = system.graph.translation_group()
                sample = system.hilbert_space.random_state(jax.random.PRNGKey(0))
                translated_samples = translation_group @ sample
                logpsi_q = net.network.apply(vars, translated_samples)
                psi_q = jnp.exp(logpsi_q)
                abs_psi = jnp.abs(psi_q)
                arg_psi = jnp.angle(psi_q)
                positions = patches.compute_positions()
                phases = positions @ jnp.array(q) % (2 * jnp.pi)
                phases = jnp.floor(phases / (2 * jnp.pi)) * 2 * jnp.pi  # round down,
                # to avoid cases where phase is just under 2pi due to being float
                print("|z| = ", abs_psi)
                print("arg(z) = ", arg_psi)
                print("q.r % 2pi = ", phases)
                # Check properties of output are correct
                # 1. Absolute values are all the same
                assert jnp.allclose(abs_psi, abs_psi[0])
                unique_args = jnp.unique(jnp.round(arg_psi, decimals=10))
                # 2. The phase corresponds to q*r
                for unique_arg in unique_args:
                    arg_indices = jnp.isclose(unique_arg, arg_psi)
                    print(
                        "isclose",
                        jnp.isclose(phases[arg_indices][0], phases[arg_indices]),
                    )
                    print(
                        phases[arg_indices][
                            jnp.invert(
                                jnp.isclose(phases[arg_indices][0], phases[arg_indices])
                            )
                        ]
                    )
                    assert jnp.allclose(phases[arg_indices][0], phases[arg_indices])


@pytest.mark.parametrize(
    "system",
    (BCCHeisenberg(lattice_shape=(4, 2, 2), J=(1,)),),
)
def test_VanillaHead(system):
    # Check translational invariance
    q = system.graph.ndim * (0,)
    net = ViTNd(
        depth=2,
        d_model=12,
        heads=6,
        output_head="Vanilla",
        expansion_factor=2,
        q=q,  # dummy,
        sign_net=False,
        system=system,
    )

    vars = net.network.init(
        jax.random.PRNGKey(0),
        system.hilbert_space.random_state(jax.random.PRNGKey(0), size=1),
    )
    translation_group = system.graph.translation_group()
    sample = system.hilbert_space.random_state(jax.random.PRNGKey(0))
    translated_samples = translation_group @ sample
    logpsis = net.network.apply(vars, translated_samples)
    assert jnp.allclose(logpsis[0], logpsis)
