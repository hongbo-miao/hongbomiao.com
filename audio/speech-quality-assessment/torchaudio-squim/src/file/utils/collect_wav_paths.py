from pathlib import Path


def collect_wav_paths(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(target.glob("*.wav"))
    if target.is_file():
        return [target]
    return []
