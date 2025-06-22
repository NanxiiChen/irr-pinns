from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal, constant, zeros, uniform


class FourierEmbedding(nn.Module):
    emb_scale: float = 2.0
    emb_dim: int = 64

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel",
            normal(self.emb_scale),
            (x.shape[-1], self.emb_dim),
        )
        return jnp.concatenate(
            [
                jnp.sin(jnp.pi * jnp.dot(x, kernel)),
                jnp.cos(jnp.pi * jnp.dot(x, kernel)),
            ],
            axis=-1,
        )


class RBFEmbedding(nn.Module):
    emb_dim: int = 64
    emb_scale: float = 0.1
    emb_width: float = 0.05

    @nn.compact
    def __call__(self, x):

        centers = self.param(
            "kernel",
            normal(self.emb_scale),
            (self.emb_dim, x.shape[-1]),
        )  # --> shape (emb_dim, xdim)

        x = jnp.expand_dims(x, axis=0)
        dist_sq = jnp.sum((x - centers) ** 2, axis=-1)
        rbf = jnp.exp(-dist_sq / (2 * self.emb_width**2))
        return rbf


class ExponentialEmbedding(nn.Module):
    emb_scale: float = 2.0
    emb_dim: int = 32

    @nn.compact
    def __call__(self, x):
        low, high = 0, self.emb_scale

        def kernel_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype=dtype, minval=low, maxval=high)

        kernel = self.param("kernel", kernel_init, (x.shape[-1], self.emb_dim))
        x_proj = jnp.dot(x, kernel)
        embedded = jnp.exp(x_proj)
        
        return embedded
    

class WaveletEmbedding(nn.Module):
    emb_dim: int = 64
    emb_scale: float = 1.0
    wavelet_type: str = "mexican_hat"
    scale_range: tuple = (0.1, 2.0)  # 尺度范围

    @nn.compact
    def __call__(self, x):
        # 使用对数空间，优化更稳定
        log_scale_min = jnp.log(self.scale_range[0])
        log_scale_max = jnp.log(self.scale_range[1])
        
        log_scales = self.param(
            "log_scales",
            lambda key, shape: jax.random.uniform(
                key, shape, 
                minval=log_scale_min, 
                maxval=log_scale_max
            ),
            (self.emb_dim, x.shape[-1]),
        )
        
        scales = jnp.exp(log_scales)  # 转换回真实尺度
        
        translations = self.param(
            "translations",
            normal(self.emb_scale),
            (self.emb_dim, x.shape[-1]),
        )
        
        
        # 扩展输入维度用于广播
        x_expanded = jnp.expand_dims(x, axis=-2)  # shape: (..., 1, input_dim)
        
        # 计算小波变换
        if self.wavelet_type == "morlet":
            # Morlet小波：适合捕捉振荡性局部特征
            omega0 = 6.0  # 中心频率参数
            normalized_x = (x_expanded - translations) / scales
            wavelet_real = jnp.exp(-0.5 * normalized_x**2) * jnp.cos(omega0 * normalized_x)
            wavelet_imag = jnp.exp(-0.5 * normalized_x**2) * jnp.sin(omega0 * normalized_x)
            
        elif self.wavelet_type == "mexican_hat":
            # Mexican Hat小波：适合检测边缘和突变
            normalized_x = (x_expanded - translations) / scales
            wavelet_real = (1 - normalized_x**2) * jnp.exp(-0.5 * normalized_x**2)
            # wavelet_imag = jnp.zeros_like(wavelet_real)
            
        elif self.wavelet_type == "gabor":
            # Gabor小波：结合局部化和频率选择性
            sigma = 0.5
            normalized_x = (x_expanded - translations) / scales
            envelope = jnp.exp(-0.5 * (normalized_x / sigma)**2)
            wavelet_real = envelope * jnp.cos(2 * jnp.pi * normalized_x)
            wavelet_imag = envelope * jnp.sin(2 * jnp.pi * normalized_x)
        
        # 计算小波系数的模值（更稳定）
        # wavelet_magnitude = jnp.sqrt(wavelet_real**2 + wavelet_imag**2)
        
        # 沿空间维度求和得到特征
        features_real = jnp.sum(wavelet_real, axis=-1)  # shape: (..., emb_dim//2)
        # features_imag = jnp.sum(wavelet_imag, axis=-1)
        
        # return jnp.concatenate([features_real, features_imag], axis=-1)
        return features_real