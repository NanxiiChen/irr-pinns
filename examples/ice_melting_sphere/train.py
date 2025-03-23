from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import jit


def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    params = model.init(rng, jnp.ones(3), jnp.ones(1))
    scheduler = optax.exponential_decay(lr, decay_every, decay, staircase=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(scheduler),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@partial(jit, static_argnums=(3,))
def train_step(state, batch, eps, loss_fn):
    params = state.params
    (weighted_loss, (loss_components, weight_components, aux)), grads = (
        jax.value_and_grad(loss_fn, has_aux=True, argnums=0)(params, batch, eps)
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux)
