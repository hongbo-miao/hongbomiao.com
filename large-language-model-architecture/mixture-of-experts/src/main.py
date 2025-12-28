import logging

import torch
import torch.nn.functional as F  # noqa: N812
from models.mixture_of_experts import MixtureOfExperts
from utils.generate_synthetic_data import generate_synthetic_data

logger = logging.getLogger(__name__)


def main() -> None:
    input_dimension = 32
    hidden_dimension = 64
    output_dimension = 32
    expert_count = 4
    top_k = 2

    batch_size = 16
    sequence_length = 10
    epoch_count = 100
    learning_rate = 0.001

    train_sample_count = 960  # 960 / 16 = 60 batches per epoch
    test_sample_count = 192

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    logger.info(f"Using device: {device}")

    logger.info("Generating synthetic data...")
    target_transform = torch.randn(input_dimension, output_dimension) * 0.1
    train_inputs, train_targets = generate_synthetic_data(
        train_sample_count,
        sequence_length,
        input_dimension,
        target_transform,
    )
    test_inputs, test_targets = generate_synthetic_data(
        test_sample_count,
        sequence_length,
        input_dimension,
        target_transform,
    )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    model = MixtureOfExperts(
        input_dimension=input_dimension,
        hidden_dimension=hidden_dimension,
        output_dimension=output_dimension,
        expert_count=expert_count,
        top_k=top_k,
        load_balance_weight=0.01,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"Model parameters: {total_parameters:,}")
    logger.info(f"Expert count: {expert_count}, Top-k: {top_k}")
    logger.info(
        f"Training samples: {train_sample_count}, Test samples: {test_sample_count}",
    )
    logger.info("Starting training...")

    for epoch in range(epoch_count):
        model.train()
        epoch_task_loss = 0.0
        epoch_balance_loss = 0.0
        batch_count = 0

        permutation = torch.randperm(train_sample_count, device=device)

        for batch_start in range(0, train_sample_count, batch_size):
            batch_indices = permutation[batch_start : batch_start + batch_size]
            batch_inputs = train_inputs[batch_indices]
            batch_targets = train_targets[batch_indices]

            optimizer.zero_grad()

            predictions, load_balance_loss = model(batch_inputs)

            task_loss = F.mse_loss(predictions, batch_targets)
            total_loss = task_loss + load_balance_loss

            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_balance_loss += load_balance_loss.item()
            batch_count += 1

        average_task_loss = epoch_task_loss / batch_count
        average_balance_loss = epoch_balance_loss / batch_count

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_predictions, _ = model(test_inputs)
                test_loss = F.mse_loss(test_predictions, test_targets).item()

            logger.info(
                f"Epoch {epoch + 1:3d}/{epoch_count} | "
                f"Train Loss: {average_task_loss:.4f} | "
                f"Balance Loss: {average_balance_loss:.4f} | "
                f"Test Loss: {test_loss:.4f}",
            )

    model.eval()
    with torch.no_grad():
        final_predictions, _ = model(test_inputs)
        final_test_loss = F.mse_loss(final_predictions, test_targets).item()

        router_logits = model.router_weight(test_inputs.view(-1, input_dimension))
        router_probabilities = F.softmax(router_logits, dim=-1)
        expert_selection_count = torch.zeros(expert_count, device=device)
        _, top_k_indices = torch.topk(router_probabilities, top_k, dim=-1)
        for expert_index in range(expert_count):
            expert_selection_count[expert_index] = (
                (top_k_indices == expert_index).sum().item()
            )

    logger.info("-" * 60)
    logger.info(f"Final test loss: {final_test_loss:.4f}")
    logger.info("Expert utilization on test set:")
    total_selections = expert_selection_count.sum().item()
    for expert_index in range(expert_count):
        percentage = (
            100.0 * expert_selection_count[expert_index].item() / total_selections
        )
        logger.info(f"  Expert {expert_index}: {percentage:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
