import logging

import torch
from qwen_asr import Qwen3ASRModel

logger = logging.getLogger(__name__)


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map=device,
        # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM
        max_inference_batch_size=32,
        # Maximum number of tokens to generate. Set a larger value for long audio input
        max_new_tokens=256,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs={
            "dtype": torch.bfloat16,
            "device_map": device,
        },
    )

    results = model.transcribe(
        audio=[
            "data/audio.wav",
        ],
        language=None,
        return_time_stamps=True,
    )

    for result in results:
        logger.info(f"Language: {result.language}")
        logger.info(f"Text: {result.text}")
        logger.info("Timestamps:")
        for timestamp in result.time_stamps:
            logger.info(f"  {timestamp}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
