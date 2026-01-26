from flax import linen as nn
from netket.utils.types import Array
import jax
import jax.numpy as jnp
import netket as nk


class FlipExpSum(nn.Module):
    """
    Symmetrize wavefunction according to  \Psi_symm(x) = 0.5*(\Psi(x) + coeff*\Psi(-x))
    Implemented as logsumexp(), returning log(\Psi_symm(x)) and assuming log(\Psi(x)) is returned by self.module
    """

    module: nn.Module
    symmetrize: bool = True

    def setup(self):
        if self.symmetrize:
            self.coeff = 1.0
            self.logsumexp = jax.scipy.special.logsumexp
        else:
            self.coeff = -1.0
            self.logsumexp = nk.jax.logsumexp_cplx

    def __call__(self, x: Array) -> Array:
        logpsi_plus = self.module(x)
        logpsi_minus = self.module(-x)
        logpsi_pm = jnp.stack(
            (logpsi_plus, logpsi_minus), axis=0
        )  # stack along a new zero axis, to be summed over
        characters = jnp.array([1.0, self.coeff], dtype=x.dtype)
        characters = jnp.expand_dims(characters, tuple(range(1, x.ndim)))
        logpsi = self.logsumexp(
            logpsi_pm, axis=0, b=characters / 2.0
        )  # compute log(0.5*(exp(logpsi_plus)+exp(logpsi_minus)))
        return logpsi
