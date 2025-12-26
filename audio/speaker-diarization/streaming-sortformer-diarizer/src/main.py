import logging
from pathlib import Path

import torch
from nemo.collections.asr.models import SortformerEncLabelModel
from shared.services.log_diarization_segments import log_diarization_segments

logger = logging.getLogger(__name__)

AUDIO_PATH = Path("data/audio.wav")
MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    logger.info(f"Using device: {device}")

    logger.info(f"Loading pretrained Sortformer diarization model: {MODEL_NAME}")
    diar_model = SortformerEncLabelModel.from_pretrained(MODEL_NAME)
    diar_model.to(device)
    diar_model.eval()

    predicted_segments = diar_model.diarize(audio=[str(AUDIO_PATH)], batch_size=1)
    diarization_segments = predicted_segments[0]
    log_diarization_segments(diarization_segments)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
