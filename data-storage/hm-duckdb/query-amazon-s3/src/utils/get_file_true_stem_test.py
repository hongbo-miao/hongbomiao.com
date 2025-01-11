from pathlib import Path

from utils.get_file_true_stem import get_file_true_stem


class TestGetFileTrueStem:
    def test_file_path(self) -> None:
        file_path = Path("/path/to/my.txt")
        assert get_file_true_stem(file_path) == "my"

    def test_file_name(self) -> None:
        file_path = Path("my.txt")
        assert get_file_true_stem(file_path) == "my"

    def test_multiple_dot_extension(self) -> None:
        file_path = Path("/path/to/my.zstd.parquet")
        assert get_file_true_stem(file_path) == "my"

    def test_hidden_file_path(self) -> None:
        file_path = Path("/path/to/.my.txt")
        assert get_file_true_stem(file_path) == ".my"

    def test_hidden_file_name_with_extension(self) -> None:
        file_path = Path(".my.txt")
        assert get_file_true_stem(file_path) == ".my"

    def test_hidden_file_name(self) -> None:
        file_path = Path(".my")
        assert get_file_true_stem(file_path) == ".my"

    def test_dir_file_path(self) -> None:
        file_path = Path("/path/to/my")
        assert get_file_true_stem(file_path) == "my"
