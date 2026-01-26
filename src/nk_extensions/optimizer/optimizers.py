import optax
from optax._src.base import ScalarOrSchedule, GradientTransformation, Schedule
from optax._src.transform import ScaleByScheduleState
from optax._src import numerics
from optax._src import combine
from optax._src.base import init_empty_state, identity
import jax.numpy as jnp
import jax


def norm_clipped_scale_by_schedule(
    learning_rate_fn: Schedule, sqrt_norm_constraint: float
) -> GradientTransformation:
    def init_fn(params):
        del params
        return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        update_norm = optax.global_norm(updates)
        learning_rate = learning_rate_fn(state.count)
        lr_sign = jnp.sign(learning_rate)
        step_size = lr_sign * jnp.min(
            jnp.abs(jnp.array([learning_rate, sqrt_norm_constraint / update_norm]))
        )
        updates = jax.tree.map(
            lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates
        )
        return updates, ScaleByScheduleState(count=numerics.safe_increment(state.count))

    return GradientTransformation(init_fn, update_fn)


def norm_clipped_scale(
    learning_rate: float, sqrt_norm_constraint: float
) -> GradientTransformation:
    def update_fn(updates, state, params=None):
        del params
        update_norm = optax.global_norm(updates)
        lr_sign = jnp.sign(learning_rate)
        step_size = lr_sign * jnp.min(
            jnp.abs(jnp.array([learning_rate, sqrt_norm_constraint / update_norm]))
        )
        updates = jax.tree.map(
            lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates
        )
        return updates, state

    return GradientTransformation(init_empty_state, update_fn)


def scale_by_learning_rate_clipped_norm(
    learning_rate: ScalarOrSchedule, norm_constraint: float, *, flip_sign: bool = True
) -> GradientTransformation:
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return norm_clipped_scale_by_schedule(
            lambda count: m * learning_rate(count), m * jnp.sqrt(norm_constraint)
        )
    return norm_clipped_scale(m * learning_rate, m * jnp.sqrt(norm_constraint))


def sgd_norm_clipped(
    learning_rate: ScalarOrSchedule, norm_constraint: float
) -> GradientTransformation:
    return combine.chain(
        identity(), scale_by_learning_rate_clipped_norm(learning_rate, norm_constraint)
    )
