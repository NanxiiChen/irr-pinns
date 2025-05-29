from typing import Callable

import jax.numpy as jnp
from flax import linen as nn



class ModifiedReLU(nn.Module):
    """Dense layer with modified ReLU: z_{k+1} = max(0, m_k * (W_k * z_k + b_k))"""
    

    m_init: float = 1.0
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x):
        # 标准的线性变换 W_k * z_k + b_k
        x = nn.Dense(features=x.shape[-1],
                     use_bias=self.use_bias)(x)
        
        # 可学习的系数 m_k
        m_k = self.param('m_k',
                        lambda rng, shape: jnp.full(shape, self.m_init),
                        (1,))  # 每层一个系数
        
        # 应用修改版ReLU
        return jnp.maximum(0, m_k * x)
    

class Snake(nn.Module):
    init_alpha: float = 5.0

    @nn.compact
    def __call__(self, x):
        alpha = self.param("alpha", lambda key: jnp.ones((1,)) * self.init_alpha)
        return x + (1.0 / alpha) * jnp.sin(alpha * x)**2
