from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, tree_util


def create_gradient_mask(params, freeze_model=False, freeze_eigen=False, eigen_tag="sl", model_tag="Dense"):
    def mask_fn(path, param):
        if freeze_model and model_tag in path:
            return jnp.zeros_like(param)
        if freeze_eigen and eigen_tag in path:
            return jnp.zeros_like(param)
        return jnp.ones_like(param)

    return tree_util.tree_map_with_path(mask_fn, params)


@partial(jit, static_argnums=(0, 4, 5))
def train_step(loss_fn, state, batch, eps, freeze_model, freeze_eigen, **kwargs):
    params = state.params
    
    (weighted_loss, (loss_components, weight_components, aux_vars)), grads = (
        jax.value_and_grad(loss_fn, has_aux=True)(params, batch, eps)
    )
    
    model_tag = kwargs.get("model_tag", "Dense")
    eigen_tag = kwargs.get("eigen_tag", "sl")
    grad_mask = create_gradient_mask(grads, freeze_model=freeze_model, freeze_eigen=freeze_eigen,
                                     model_tag=model_tag, eigen_tag=eigen_tag)
    masked_grads = tree_util.tree_map(lambda g, m: g * m, grads, grad_mask)
    
    new_state = state.apply_gradients(grads=masked_grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)
