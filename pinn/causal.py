import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt


class CausalWeightor:
    def __init__(
        self,
        num_chunks: int,
        t_range: tuple,
        **kwargs,
    ):

        self.num_chunks = num_chunks
        self.t_range = t_range
        # self.bins = jnp.linspace(t_range[0], t_range[1], num_chunks + 1)
        self.bins = kwargs.get(
            "bins", jnp.linspace(t_range[0], t_range[1], num_chunks + 1)
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_causal_weight(
        self, loss_chunks: jnp.array, eps: jnp.array
    ):
        cumulative_loss = jnp.cumsum(loss_chunks[:-1])

        weights = jnp.concatenate([jnp.array([1.0]), jnp.exp(-eps * cumulative_loss)])

        return jax.lax.stop_gradient(weights)

    @partial(jax.jit, static_argnums=(0,))
    def compute_causal_loss(
        self,
        residuals: jnp.array,
        t: jnp.array,
        phi: jnp.array,
        eps: jnp.array,
    ):
        
        def compute_causal_weight_single(
            residuals: jnp.array,
            data: jnp.array,
            eps: jnp.array,
        ):
            indices = jnp.digitize(data.flatten(), self.bins) - 1

            sum_residuals_sq = jax.ops.segment_sum(
                residuals**2, indices, num_segments=self.num_chunks
            )
            count_residuals = jax.ops.segment_sum(
                jnp.ones_like(residuals), indices, num_segments=self.num_chunks
            )
            loss_chunks = sum_residuals_sq / (count_residuals + 1e-12)

            causal_weights = self.compute_causal_weight(
                loss_chunks, eps
            )
            residual_weights = causal_weights[indices]

            return residual_weights, causal_weights, loss_chunks

        residual_weights_t, causal_weights_t, loss_chunks_t = compute_causal_weight_single(
            residuals, t, eps
        )
        residual_weights_phi, causal_weights_phi, loss_chunks_phi = compute_causal_weight_single(
            residuals, phi, eps
        )
        # causal_loss = jnp.dot(loss_chunks, causal_weights)
        causal_loss = jnp.dot(residuals**2, residual_weights_phi * residual_weights_t)
        return causal_loss, {
            "causal_weights": causal_weights_phi * causal_weights_t,
            "loss_chunks": (loss_chunks_phi + loss_chunks_t) / 2,
        }

    def plot_causal_info(self, causal_weights, loss_chunks, eps):

        bins = (self.bins[1:] + self.bins[:-1]) / 2

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        ax = axes[0]
        ax.plot(bins, causal_weights, marker="o")
        ax.set(
            xlabel="Time chunks",
            ylabel="Causal weights",
        )

        ax = axes[1]
        ax.plot(bins, loss_chunks, marker="o")
        ax.set(
            xlabel="Time chunks",
            ylabel="Causal loss",
        )

        fig.suptitle(f"EPS: {eps:.2e}")

        return fig

    def update_causal_eps(
        self,
        eps,
        causal_weight,
        causal_configs,
    ):
        new_eps = eps
        if (
            causal_weight[-1] > causal_configs["max_last_weight"]
            and eps < causal_configs["max_eps"]
        ):
            # new_causal_configs["eps"] = (
            #     causal_configs["eps"] * causal_configs["step_size"]
            # )
            # print(f"Inc. eps to {causal_configs['eps']}")

            new_eps = eps * causal_configs["step_size"]
            print(f"Inc. eps to {new_eps}")

        if jnp.mean(causal_weight) < causal_configs["min_mean_weight"]:
            # new_causal_configs["eps"] = (
            #     causal_configs["eps"] / causal_configs["step_size"]
            # )
            # print(f"Dec. eps to {causal_configs['eps']}")
            new_eps = eps / causal_configs["step_size"]
            print(f"Dec. eps to {new_eps}")

        return new_eps


if __name__ == "__main__":
    t = jnp.array([0.0, 0.1, 0.7, 0.9, 0.3, 0.5, 1.0]).reshape(-1, 1)
    t_range = (0.0, 1.0)
    num = 4
    weightor = CausalWeightor(num, t_range)
    print(weightor._split_t(t))
