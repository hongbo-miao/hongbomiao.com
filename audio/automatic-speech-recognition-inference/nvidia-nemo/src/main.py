import logging
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch

logger = logging.getLogger(__name__)

AUDIO_PATH = Path("data/audio.wav")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading pretrained ASR model...")
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name="stt_en_conformer_transducer_large",
    )
    asr_model.to(device)
    asr_model.eval()

    # https://github.com/NVIDIA-NeMo/NeMo/issues/15145#issuecomment-3630162216
    asr_model.decoding.decoding.decoding_computer.disable_cuda_graphs()

    logger.info("Transcribing audio file...")
    transcriptions = asr_model.transcribe(
        [str(AUDIO_PATH)],
        batch_size=1,
    )

    if not transcriptions:
        logger.error("Transcription failed: No results returned")
        return

    hypothesis = transcriptions[0]
    text = hypothesis.text
    logger.info(f"ASR Output: {text}")

    word_count = len(text.split())
    character_count = len(text)
    logger.info(f"Word count: {word_count}")
    logger.info(f"Character count: {character_count}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
