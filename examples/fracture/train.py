from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, tree_util


def create_gradient_mask(params, freeze: bool, tag: str="phi"):
    def mask_fn(path, param):
        if freeze and tag in "/".join([str(key.key) for key in path]):
            return jnp.zeros_like(param)
        return jnp.ones_like(param)

    return tree_util.tree_map_with_path(mask_fn, params)


@partial(jit, static_argnums=(0, 4, 5))
def train_step(loss_fn, state, batch, eps, freeze, tag, **kwargs):
    params = state.params
    
    (weighted_loss, (loss_components, weight_components, aux_vars)), grads = (
        jax.value_and_grad(loss_fn, has_aux=True)(params, batch, eps)
    )
    
    grad_mask = create_gradient_mask(grads, freeze, tag)
    masked_grads = tree_util.tree_map(lambda g, m: g * m, grads, grad_mask)
    
    new_state = state.apply_gradients(grads=masked_grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)
