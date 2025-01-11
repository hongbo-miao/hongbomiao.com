from pathlib import Path


def get_file_true_stem(file_path: Path) -> str:
    stem = file_path.stem
    return stem if stem == str(file_path) else get_file_true_stem(Path(stem))
