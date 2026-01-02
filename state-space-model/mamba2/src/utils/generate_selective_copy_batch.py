import torch


def generate_selective_copy_batch(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    pad_token: int,
    copy_token: int,
    vocab_start: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of selective copying examples.

    Returns input sequences and target sequences.
    Target is the token after COPY marker, PAD elsewhere.
    """
    inputs = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    targets = torch.full((batch_size, sequence_length), pad_token, dtype=torch.long)

    for batch_index in range(batch_size):
        position = 0
        while position < sequence_length - 1:
            # Randomly insert COPY marker followed by a token to copy
            if torch.rand(1).item() < 0.3 and position < sequence_length - 2:
                inputs[batch_index, position] = copy_token
                token_to_copy = torch.randint(vocab_start, vocab_size, (1,)).item()
                inputs[batch_index, position + 1] = token_to_copy
                targets[batch_index, position + 1] = token_to_copy
                position += 2
            else:
                # Random non-COPY token
                inputs[batch_index, position] = torch.randint(
                    vocab_start,
                    vocab_size,
                    (1,),
                ).item()
                position += 1

    return inputs.to(device), targets.to(device)
