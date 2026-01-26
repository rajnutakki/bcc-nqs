from vmc.system import BCCHeisenberg
from nets.blocks.sign import SignHelper, SignRule, SignNet
from netket.models import RBM
import jax.numpy as jnp
import jax
import pytest

test_input = [
    pytest.param(
        BCCHeisenberg(lattice_shape=(4, 2, 2), J=(1, 2), sign_rule=1),
        (0,),  # sublattices
        jnp.array(
            [[1] + [1] + [1] + 29 * [-1], [-1] + [-1] + 30 * [1]]  # nx = 2, #nx = 15
        ),
        jnp.array([2, 15]),
        id="BCC J1 sign rule",
    ),
]


@pytest.mark.parametrize("system, sublattices, samples, expected_output", test_input)
def test_signhelper(system, sublattices, samples, expected_output):
    sh = SignHelper(system.graph, sublattices)
    nx = sh.compute_nx(samples)
    assert jnp.allclose(nx, expected_output)


@pytest.mark.parametrize("system, sublattices, samples, expected_output", test_input)
def test_signrule(system, sublattices, samples, expected_output):
    sh = SignHelper(system.graph, sublattices)
    signrule = SignRule(sh.compute_nx)
    vars = signrule.init(jax.random.PRNGKey(0), samples)
    res = signrule.apply(vars, samples)
    assert jnp.allclose(res, (-1) ** sh.compute_nx(samples))


@pytest.mark.parametrize("system, sublattices, samples, expected_output", test_input)
def test_signnet(system, sublattices, samples, expected_output):
    sh = SignHelper(system.graph, sublattices)
    signrule = SignRule(sh.compute_nx)
    vars = signrule.init(jax.random.PRNGKey(0), samples)
    signs = signrule.apply(vars, samples)
    net = RBM(alpha=2)
    sgn_net = SignNet(net, SignRule, sh.compute_nx, dtype=jnp.complex128)
    vars = sgn_net.init(jax.random.PRNGKey(0), samples)
    vars_rbm = {"params": vars["params"]["logpsi"]}
    res_sgn = sgn_net.apply(vars, samples)
    res_RBM = net.apply(vars_rbm, samples)
    assert jnp.allclose(signs * jnp.exp(res_sgn), jnp.exp(res_RBM))
