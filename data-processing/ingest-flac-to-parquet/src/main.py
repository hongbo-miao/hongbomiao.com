import logging
from pathlib import Path

import numpy as np
import polars as pl
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


def convert_flac_to_parquet(audio_dir: Path, parquet_path: Path) -> None:
    audio_records: list[dict[str, str | int | list[float]]] = []

    audio_file_paths = sorted(audio_dir.glob("*.flac"))

    for file_path in audio_file_paths:
        data, sample_rate = sf.read(file_path)

        # Convert stereo/multi-channel audio to mono by averaging across channels
        if data.ndim > 1:
            data = data.mean(axis=1)

        sample = data.tolist()
        audio_records.append(
            {
                "file_name": file_path.name,
                "sample_rate": sample_rate,
                "sample": sample,
            },
        )

    df = pl.DataFrame(audio_records)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path, compression="zstd", compression_level=19)
    logger.info(f"Saved audio data to {parquet_path}")


def play_audios(parquet_path: Path) -> None:
    df = pl.read_parquet(parquet_path)

    for file_name, sample_rate, samples in df.iter_rows():
        logger.info(f"Playing {file_name} at {sample_rate} Hz")
        samples_array = np.array(samples, dtype=np.float32)

        sd.play(samples_array, samplerate=sample_rate)
        sd.wait()


def main() -> None:
    audio_dir = Path("data")
    parquet_file = Path("output/audio_data.parquet")

    convert_flac_to_parquet(audio_dir=audio_dir, parquet_path=parquet_file)
    play_audios(parquet_file)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
