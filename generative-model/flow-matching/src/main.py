import logging

import torch
from shared.services.flow_matching_model import FlowMatchingModel
from shared.services.train_flow_matching import train_flow_matching
from shared.utils.generate_samples import generate_samples
from shared.utils.generate_two_moons_data import generate_two_moons_data
from shared.utils.get_device import get_device
from shared.utils.visualize_results import visualize_results
from torch import nn

logger = logging.getLogger(__name__)

SAMPLE_COUNT = 1000
EPOCH_COUNT = 10000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
HIDDEN_DIMENSION = 128
TIME_STEP_COUNT = 100


def main() -> None:
    """
    Run flow matching training and sampling.

    Flow Matching learns a velocity field v(x, t) that transforms samples from
    a simple prior distribution (Gaussian) to a target data distribution.
    Key concepts:
    1. We define a path x_t = (1-t)*x_0 + t*x_1 from noise x_0 to data x_1
    2. The velocity along this path is: dx_t/dt = x_1 - x_0
    3. We train a neural network to predict this velocity given (x_t, t)
    4. At inference, we integrate the learned velocity field to generate samples.
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    target_data = generate_two_moons_data(sample_count=SAMPLE_COUNT).to(device)
    logger.info(f"Generated {SAMPLE_COUNT} two-moons samples")

    model = FlowMatchingModel(
        input_dimension=2,
        hidden_dimension=HIDDEN_DIMENSION,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    train_flow_matching(
        model=model,
        target_data=target_data,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch_count=EPOCH_COUNT,
        batch_size=BATCH_SIZE,
    )

    generated_samples = generate_samples(
        model=model,
        sample_count=SAMPLE_COUNT,
        time_step_count=TIME_STEP_COUNT,
        device=device,
    )

    visualize_results(
        target_data=target_data.cpu(),
        generated_samples=generated_samples.cpu(),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
