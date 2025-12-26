import logging

import torch
from models.liquid_neural_network import LiquidNeuralNetwork
from torch import nn
from utils.generate_sine_wave_data import generate_sine_wave_data
from utils.train_liquid_neural_network import train_liquid_neural_network

logger = logging.getLogger(__name__)


def main() -> None:
    input_size = 1
    hidden_size = 32
    output_size = 1
    sequence_length = 50
    sample_count = 100
    epoch_count = 500
    learning_rate = 0.01

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    logger.info(f"Using device: {device}")

    logger.info("Generating sine wave data...")
    input_data, target_data = generate_sine_wave_data(sample_count, sequence_length)
    input_data = input_data.to(device)
    target_data = target_data.to(device)
    logger.info(f"Input shape: {input_data.shape}, Target shape: {target_data.shape}")

    logger.info("Creating Liquid Neural Network model...")
    model = LiquidNeuralNetwork(input_size, hidden_size, output_size).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"Model parameter count: {parameter_count}")

    logger.info("Training Liquid Neural Network...")
    train_liquid_neural_network(
        model,
        input_data,
        target_data,
        epoch_count,
        learning_rate,
    )

    logger.info("Evaluating model...")
    model.eval()
    with torch.no_grad():
        prediction = model(input_data)
        criterion = nn.MSELoss()
        final_loss = criterion(prediction, target_data)
        logger.info(f"Final evaluation loss: {final_loss.item():.6f}")

    logger.info("Liquid Neural Network training completed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
