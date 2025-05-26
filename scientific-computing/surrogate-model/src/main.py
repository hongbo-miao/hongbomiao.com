# Kriging surrogate model
import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from smt.sampling_methods import LHS  # Latin Hypercube Sampling
from smt.surrogate_models import KRG  # Kriging model

logger = logging.getLogger(__name__)


# 1. Define the true function (the function we want to approximate)
# This is a simple 1D function.
def true_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x * 6 - 2) ** 2 * np.sin(x * 12 - 4)


def main() -> None:
    # 2. Generate training data
    # We'll use Latin Hypercube Sampling (LHS) to get well-distributed points.
    # Define the bounds for our input variable x
    # Single input variable, ranging from 0 to 1
    xlimits = np.array([[0.0, 1.0]])

    # Number of training points
    n_train_points = 12

    # Initialize LHS sampler
    sampling = LHS(xlimits=xlimits, criterion="maximin", random_state=42)

    # Generate training input points
    x_train = sampling(n_train_points)

    # Evaluate the true function at these points to get training output
    y_train = true_function(x_train)

    # 3. Initialize and train the Kriging surrogate model
    # SMT offers various surrogate models. KRG is for Kriging.
    # theta0 is an initial guess for hyperparameters
    model = KRG(theta0=[1e-2], print_prediction=False, print_training=False)

    # Set training data
    model.set_training_values(x_train, y_train)

    # Train the model
    logger.info("Training the Kriging model...")
    model.train()
    logger.info("Training complete.")

    # 4. Make predictions using the trained model
    # Generate a denser set of points for prediction and plotting
    n_predict_points: int = 100
    x_predict: NDArray[np.float64] = np.linspace(
        xlimits[0, 0],
        xlimits[0, 1],
        n_predict_points,
    ).reshape(-1, 1)

    # Predict y values using the surrogate model
    y_predict_surrogate: NDArray[np.float64] = model.predict_values(x_predict)

    # Optionally, predict variance (confidence interval) if needed
    y_predict_variance: NDArray[np.float64] = model.predict_variances(x_predict)

    # For comparison, calculate the true function values at these prediction points
    y_predict_true: NDArray[np.float64] = true_function(x_predict)

    # 5. Visualize the results
    logger.info("Plotting results...")
    plt.figure(figsize=(12, 7))

    # Plot the true function
    plt.plot(x_predict, y_predict_true, "k-", label="True Function", linewidth=2)

    # Plot the surrogate model's predictions
    plt.plot(
        x_predict,
        y_predict_surrogate,
        "b--",
        label="Kriging Surrogate",
        linewidth=2,
    )

    # Plot the training points
    plt.scatter(
        x_train,
        y_train,
        c="r",
        s=100,
        marker="o",
        edgecolor="k",
        label="Training Points",
    )

    # Add confidence interval if variance was predicted
    if "y_predict_variance" in locals():
        std_dev: NDArray[np.float64] = np.sqrt(y_predict_variance)
        plt.fill_between(
            x_predict.ravel(),
            (y_predict_surrogate - 2 * std_dev).ravel(),
            (y_predict_surrogate + 2 * std_dev).ravel(),
            color="lightblue",
            alpha=0.5,
            label="95% Confidence Interval",
        )

    plt.xlabel("Input x", fontsize=14)
    plt.ylabel("Output y", fontsize=14)
    plt.title("Kriging Surrogate Model", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(linestyle=":", alpha=0.7)
    # Adjust y-limits for better visualization
    plt.ylim(np.min(y_predict_true) - 5, np.max(y_predict_true) + 5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
