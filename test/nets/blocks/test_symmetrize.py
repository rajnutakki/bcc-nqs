import numpy as np
import jax
import netket as nk
from nets.blocks.symmetrize import FlipExpSum


def test_flip_exp_sum_symm_and_asymm():
    net = nk.models.RBM()
    symm_net = FlipExpSum(net, symmetrize=True)
    asymm_net = FlipExpSum(net, symmetrize=False)
    x = np.array([1, 1, 1, -1])
    x = np.full((1, 4), x)
    params = symm_net.init(jax.random.PRNGKey(1234), x)
    symm_out1 = np.exp(symm_net.apply(params, x))
    symm_out2 = np.exp(symm_net.apply(params, -x))
    asymm_out1 = np.exp(asymm_net.apply(params, x))
    asymm_out2 = np.exp(asymm_net.apply(params, -x))
    np.testing.assert_allclose(
        symm_out1, symm_out2, err_msg="Symmetrized outputs should be equal for x and -x"
    )
    np.testing.assert_allclose(
        asymm_out1,
        -asymm_out2,
        err_msg="Antisymmetrized outputs should be negatives for x and -x",
    )
