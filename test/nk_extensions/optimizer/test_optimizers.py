import netket as nk
from nk_extensions.optimizer import sgd_norm_clipped
import jax
import jax.numpy as jnp
import numpy as np
import optax
from netket.driver.abstract_variational_driver import apply_gradient


def _pytree_allclose(tree_a, tree_b):
    if jax.tree.structure(tree_a) != jax.tree.structure(tree_b):
        return False

    leaves_a, leaves_b = jax.tree.leaves(tree_a), jax.tree.leaves(tree_b)
    return all(jnp.allclose(a, b) for a, b in zip(leaves_a, leaves_b))


def _setup(vstate=None):
    N = 8
    hi = nk.hilbert.Spin(1 / 2, N)
    graph = nk.graph.Chain(N)
    model = nk.models.RBM(alpha=1)
    # sampler = nk.sampler.MetropolisLocal(hi)
    ham = nk.operator.Heisenberg(hilbert=hi, graph=graph)
    if vstate is None:
        vstate = nk.vqs.FullSumState(hi, model)

    def driver_t(optimizer):
        return nk.driver.VMC(
            hamiltonian=ham,
            optimizer=optimizer,
            variational_state=vstate,
        )

    return driver_t, vstate


def test_sgd_norm_clipped():
    # Test that the VMC_NG driver runs with the sgd_norm_clipped optimizer
    driver_t, _ = _setup()
    driver = driver_t(sgd_norm_clipped(learning_rate=0.01, norm_constraint=0.01))
    driver.run(1)


def test_sgd_norm_clipped_update_smallconstraint():
    # Test the update for a small constraint, which will be clipped
    norm_constraint = 1e-5
    lr = 0.01
    driver_t, vstate = _setup()
    optimizer = sgd_norm_clipped(learning_rate=lr, norm_constraint=norm_constraint)
    driver = driver_t(optimizer)
    opt_state = optimizer.init(vstate.parameters)
    dp = driver._forward_and_backward()
    # Check that applying the gradient works
    _, _ = apply_gradient(optimizer.update, opt_state, dp, vstate.parameters)

    assert np.sqrt(norm_constraint) / optax.global_norm(dp) < lr, (
        "Gradient should be clipped"
    )
    optimizer_clipped = optax.sgd(
        np.sqrt(norm_constraint) / optax.global_norm(dp)
    )  # sgd with norm constraint instead of lr
    driver_clipped_t, vstate = _setup(vstate)
    driver_clipped = driver_clipped_t(optimizer_clipped)
    dp_clipped = driver_clipped._forward_and_backward()
    print(dp)
    print(dp_clipped)
    assert _pytree_allclose(dp, dp_clipped)


def test_sgd_norm_clipped_update_largeconstraint():
    # Test the update for a large constraint, which will not be clipped
    norm_constraint = 1e5
    lr = 0.01
    driver_t, vstate = _setup()
    optimizer = sgd_norm_clipped(learning_rate=lr, norm_constraint=norm_constraint)
    driver = driver_t(optimizer)
    opt_state = optimizer.init(vstate.parameters)
    dp = driver._forward_and_backward()
    # Check that applying the gradient works
    _, _ = apply_gradient(optimizer.update, opt_state, dp, vstate.parameters)

    assert np.sqrt(norm_constraint) / optax.global_norm(dp) > lr, (
        "Gradient should not be clipped"
    )
    optimizer_sgd = optax.sgd(lr)  # sgd with norm constraint instead of lr
    driver_sgd_t, vstate = _setup(vstate)
    driver_sgd = driver_sgd_t(optimizer_sgd)
    dp_sgd = driver_sgd._forward_and_backward()
    print(dp)
    print(dp_sgd)
    assert _pytree_allclose(dp, dp_sgd)
