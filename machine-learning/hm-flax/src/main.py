from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def create_train_state(
    rng: jax.random.PRNGKey,
    learning_rate: float,
) -> train_state.TrainState:
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, 10)
    return optax.softmax_cross_entropy(logits, one_hot).mean()


def loss_fn(
    params: dict[str, Any],
    apply_fn: Any,
    images: jnp.ndarray,
    labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    logits = apply_fn({"params": params}, images)
    loss = cross_entropy_loss(logits, labels)
    return loss, logits


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: dict[str, jnp.ndarray],
) -> tuple[train_state.TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(
        state.params,
        state.apply_fn,
        batch["image"],
        batch["label"],
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


def main() -> None:
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate=1e-3)

    for epoch in range(10):
        batch = {
            "image": jax.random.normal(rng, (32, 28, 28, 1)),
            "label": jax.random.randint(rng, (32,), 0, 10),
        }
        state, loss = train_step(state, batch)
        print(f"Epoch {epoch}, Loss: {loss}")

    test_image = jax.random.normal(rng, (1, 28, 28, 1))
    logits = state.apply_fn({"params": state.params}, test_image)
    predicted_class = jnp.argmax(logits, axis=-1)
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
