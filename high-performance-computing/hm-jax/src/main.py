import logging

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit, random

logger = logging.getLogger(__name__)


# The linear model
def predict(w: float, b: float, X: jnp.ndarray) -> jnp.ndarray:
    return w * X + b


# The loss function (mean squared error)
def loss_fn(w: float, b: float, X: jnp.ndarray, y: jnp.ndarray) -> float:
    predictions = predict(w, b, X)
    return jnp.mean((predictions - y) ** 2)


# JIT compile the update step for efficiency
@jit
def update(
    w: float,
    b: float,
    X: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
) -> tuple[float, float]:
    dw, db = grad(loss_fn, argnums=(0, 1))(w, b, X, y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b


def main() -> None:
    # Generate synthetic data
    key = random.PRNGKey(0)
    N = 100  # number of data points
    X = random.normal(key, (N,))  # features

    # Adding even more non-linearity and varying noise
    true_w, true_b = 2.0, -1.0  # true weights for the linear model
    y = (
        true_w * X
        + true_b
        + jnp.sin(X) * 0.5
        + jnp.cos(2 * X) * 0.3  # additional non-linear component
        + random.normal(key, (N,)) * jnp.abs(X) * 0.5  # varying noise based on X
        + random.normal(key, (N,)) * 2.0  # additional noise for more randomness
    )

    # Initialize weights
    w = jnp.array(0.0)
    b = jnp.array(0.0)

    # Define the learning rate and the number of iterations
    learning_rate = 0.1
    num_iterations = 100

    # Train
    losses = []
    for i in range(num_iterations):
        w, b = update(w, b, X, y, learning_rate)
        current_loss = loss_fn(w, b, X, y)
        losses.append(current_loss)
        if i % 10 == 0:
            logger.info(
                f"Iteration {i}: loss = {current_loss:.4f}, w = {w:.4f}, b = {b:.4f}",
            )

    # Plot the results
    plt.plot(X, y, "bo", label="Data")
    plt.plot(X, predict(w, b, X), "r-", label="Fitted Line")
    plt.legend()
    plt.title(f"Linear Regression: w = {w:.2f}, b = {b:.2f}")
    plt.show()

    plt.plot(losses)
    plt.title("Loss over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
