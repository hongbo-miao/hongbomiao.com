import logging

from config import config
from diart.blocks import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.models import EmbeddingModel, SegmentationModel
from diart.sinks import RTTMWriter
from diart.sources import MicrophoneAudioSource

logger = logging.getLogger(__name__)


def main() -> None:
    segmentation = SegmentationModel.from_pretrained(
        config.SEGMENTATION_MODEL,
        config.HUGGING_FACE_HUB_TOKEN,
    )
    embedding = EmbeddingModel.from_pretrained(
        config.EMBEDDING_MODEL,
        config.HUGGING_FACE_HUB_TOKEN,
    )
    diarization_config = SpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
    )
    pipeline = SpeakerDiarization(diarization_config)
    mic = MicrophoneAudioSource()
    inference = StreamingInference(pipeline, mic, do_plot=True)
    inference.attach_observers(RTTMWriter(mic.uri, "output/audio.rttm"))
    prediction = inference()
    logger.info(prediction)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
