import logging

from diart import SpeakerDiarization
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter
from diart.sources import MicrophoneAudioSource

logger = logging.getLogger(__name__)


def main() -> None:
    pipeline = SpeakerDiarization()
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
