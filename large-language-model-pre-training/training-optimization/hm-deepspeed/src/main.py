import logging
from pathlib import Path

import deepspeed
from models.simple_neural_network import SimpleNeuralNetwork
from torch import nn
from utils.create_synthetic_dataset import create_synthetic_dataset

logger = logging.getLogger(__name__)


def main() -> None:
    input_size = 128
    hidden_size = 256
    output_size = 64
    sample_count = 1000
    batch_size = 32
    epoch_count = 5

    logger.info("Configuration:")
    logger.info(f"  Input size: {input_size}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Output size: {output_size}")
    logger.info(f"  Sample count: {sample_count}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Epoch count: {epoch_count}")

    deepspeed.init_distributed()

    logger.info("Creating model...")
    model = SimpleNeuralNetwork(input_size, hidden_size, output_size)

    logger.info("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(sample_count, input_size, output_size)

    deepspeed_config_path = Path(__file__).parent / "deepspeed_config.json"
    logger.info(f"Loading DeepSpeed config from: {deepspeed_config_path}")

    logger.info("Initializing DeepSpeed engine...")
    model_engine, _optimizer, training_dataloader, _lr_scheduler = deepspeed.initialize(
        model=model,
        config=str(deepspeed_config_path),
        training_data=dataset,
    )

    logger.info(f"DeepSpeed engine initialized on device: {model_engine.device}")

    logger.info("Starting training...")
    loss_function = nn.MSELoss()

    for epoch in range(epoch_count):
        total_loss = 0.0
        batch_count = 0

        for input_batch, target_batch in training_dataloader:
            input_tensor = input_batch.to(
                model_engine.device,
                dtype=model_engine.module.layer1.weight.dtype,
            )
            target_tensor = target_batch.to(
                model_engine.device,
                dtype=model_engine.module.layer1.weight.dtype,
            )

            output = model_engine(input_tensor)
            loss = loss_function(output, target_tensor)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / batch_count
        logger.info(
            f"Epoch {epoch + 1}/{epoch_count}, Average Loss: {average_loss:.6f}",
        )

    logger.info("Training completed successfully")

    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
