"""
Example: Train a simple Mamba model on a selective copying task.

The task: Given a sequence with special "copy" markers, the model learns to
output the tokens that follow copy markers. This demonstrates Mamba's ability
to selectively remember information based on input content.

Example:
    Input:  [A, B, COPY, C, D, COPY, E, F, PAD, PAD]
    Target: [_, _, _, C, _, _, E, _, _, _]

The model must learn to:
1. Recognize COPY markers
2. Remember and output the next token after each COPY marker
3. Ignore other tokens

"""

import logging

import torch
from models.selective_copy_model import SelectiveCopyModel
from torch import nn
from utils.generate_selective_copy_batch import generate_selective_copy_batch
from utils.get_device import get_device

logger = logging.getLogger(__name__)

PAD_TOKEN = 0
COPY_TOKEN = 1
VOCAB_START = 2
VOCAB_SIZE = 12  # PAD, COPY, and 10 data tokens


def main() -> None:
    embedding_dimension = 32
    state_dimension = 16
    head_count = 4  # inner_dim (32*2=64) must be divisible by head_count
    layer_count = 2
    sequence_length = 32
    batch_size = 32
    epoch_count = 100
    learning_rate = 1e-3

    device = get_device()

    model = SelectiveCopyModel(
        vocabulary_size=VOCAB_SIZE,
        embedding_dimension=embedding_dimension,
        state_dimension=state_dimension,
        head_count=head_count,
        layer_count=layer_count,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    logger.info(f"Training selective copy model on {device}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epoch_count):
        inputs, targets = generate_selective_copy_batch(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
            pad_token=PAD_TOKEN,
            copy_token=COPY_TOKEN,
            vocab_start=VOCAB_START,
            vocab_size=VOCAB_SIZE,
        )

        optimizer.zero_grad()
        logits = model(inputs)

        # Reshape for cross entropy: (batch * seq, vocab) vs (batch * seq,)
        loss = loss_function(
            logits.view(-1, VOCAB_SIZE),
            targets.view(-1),
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            # Calculate accuracy on copy positions only
            predictions = logits.argmax(dim=-1)
            copy_mask = targets != PAD_TOKEN
            if copy_mask.sum() > 0:
                accuracy = (predictions[copy_mask] == targets[copy_mask]).float().mean()
                logger.info(
                    f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.2%}",
                )

    # Test on a new batch
    logger.info("Testing on new batch:")
    model.eval()
    with torch.no_grad():
        test_inputs, test_targets = generate_selective_copy_batch(
            batch_size=4,
            sequence_length=sequence_length,
            device=device,
            pad_token=PAD_TOKEN,
            copy_token=COPY_TOKEN,
            vocab_start=VOCAB_START,
            vocab_size=VOCAB_SIZE,
        )
        test_logits = model(test_inputs)
        test_predictions = test_logits.argmax(dim=-1)

        for i in range(min(2, len(test_inputs))):
            logger.info(f"Example {i + 1}:")
            input_sequence = test_inputs[i].tolist()
            target_sequence = test_targets[i].tolist()
            predicted_sequence = test_predictions[i].tolist()

            # Show only positions with COPY markers
            for position, token in enumerate(input_sequence[:-1]):
                if token == COPY_TOKEN:
                    expected = target_sequence[position + 1]
                    predicted = predicted_sequence[position + 1]
                    status = "correct" if expected == predicted else "wrong"
                    logger.info(
                        f"  COPY at {position}: expected {expected}, predicted {predicted} ({status})",
                    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
