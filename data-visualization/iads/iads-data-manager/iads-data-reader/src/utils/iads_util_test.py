from pathlib import Path

import pytest
from iads_util import IadsUtil


class TestIadsDataframe:
    def test_get_irig_times(self, tmp_path: Path) -> None:
        test_file = tmp_path / "IadsArchiveInfo.txt"
        test_content = """ArchiveStartTime = 315:20:00:00.000
ArchiveEndTime = 315:20:30:00.000
Flight =
Test = Experiment1
Tail = HM
FlightDate = 01/01/25
"""
        test_file.write_text(test_content)
        irig_start_time, irig_end_time, year = IadsUtil.get_irig_times(test_file)
        assert irig_start_time == "315:20:00:00.000"
        assert irig_end_time == "315:20:30:00.000"
        assert year == 2025

    def test_get_irig_times_missing_data(self, tmp_path: Path) -> None:
        test_file = tmp_path / "IadsArchiveInfo.txt"
        test_content = """Flight =
Flight =
Test = Experiment1
Tail = HM
FlightDate = 01/01/25
"""
        test_file.write_text(test_content)
        with pytest.raises(
            ValueError,
            match="Could not find start time, end time, or year in archive info file",
        ):
            IadsUtil.get_irig_times(test_file)

    def test_get_irig_times_invalid_date_format(self, tmp_path: Path) -> None:
        test_file = tmp_path / "IadsArchiveInfo.txt"
        test_content = """ArchiveStartTime = 315:20:00:00.000
ArchiveEndTime = 315:20:30:00.000
Flight =
Test = Experiment1
Tail = HM
FlightDate = 2025-01-01
"""
        test_file.write_text(test_content)
        with pytest.raises(ValueError):
            IadsUtil.get_irig_times(test_file)

    def test_get_irig_times_invalid_time_range(self, tmp_path: Path) -> None:
        test_file = tmp_path / "IadsArchiveInfo.txt"
        test_content = """ArchiveStartTime = 315:20:30:00.000
ArchiveEndTime = 315:20:00:00.000
Flight =
Test = Experiment1
Tail = HM
FlightDate = 01/01/25
"""
        test_file.write_text(test_content)
        with pytest.raises(
            ValueError,
            match="End time '315:20:00:00.000' must be greater than start time '315:20:30:00.000'",
        ):
            IadsUtil.get_irig_times(test_file)

    def test_get_irig_times_invalid_start_time(self, tmp_path: Path) -> None:
        test_file = tmp_path / "IadsArchiveInfo.txt"
        test_content = """ArchiveStartTime = 365:00:00:00.000
ArchiveEndTime = 315:20:30:00.000
Flight =
Test = Experiment1
Tail = HM
FlightDate = 01/01/25
"""
        test_file.write_text(test_content)
        with pytest.raises(
            ValueError,
            match="Start time '365:00:00:00.000' cannot be bigger than 364:23:59:59.999",
        ):
            IadsUtil.get_irig_times(test_file)

    def test_get_irig_times_invalid_end_time_zero(self, tmp_path: Path) -> None:
        test_file = tmp_path / "IadsArchiveInfo.txt"
        test_content = """ArchiveStartTime = 315:20:00:00.000
ArchiveEndTime = 000:00:00:00.000
Flight =
Test = Experiment1
Tail = HM
FlightDate = 01/01/25
"""
        test_file.write_text(test_content)
        with pytest.raises(
            ValueError,
            match="End time '000:00:00:00.000' cannot be 000:00:00:00.000 or bigger than 364:23:59:59.999",
        ):
            IadsUtil.get_irig_times(test_file)

    def test_get_irig_times_invalid_end_time_too_big(self, tmp_path: Path) -> None:
        test_file = tmp_path / "IadsArchiveInfo.txt"
        test_content = """ArchiveStartTime = 315:20:00:00.000
ArchiveEndTime = 365:00:00:00.000
Flight =
Test = Experiment1
Tail = HM
FlightDate = 01/01/25
"""
        test_file.write_text(test_content)
        with pytest.raises(
            ValueError,
            match="End time '365:00:00:00.000' cannot be 000:00:00:00.000 or bigger than 364:23:59:59.999",
        ):
            IadsUtil.get_irig_times(test_file)

    @pytest.mark.parametrize(
        "start_time,end_time",
        [
            ("---:--:--:--.---", "---:--:--:--.---"),
            ("abc:12:34:56.789", "def:12:34:56.789"),
            ("&^%", "*()"),
            ("123:456:78:90.123", "123:45:67:89.abc"),
            ("123:45:67:89", "123:45:67:89"),  # Missing milliseconds
            ("12:34:56:78.999", "12:34:56:78.999"),  # Only 2 digits for day
        ],
    )
    def test_get_irig_times_invalid_time_format(
        self,
        tmp_path: Path,
        start_time: str,
        end_time: str,
    ) -> None:
        test_file = tmp_path / "IadsArchiveInfo.txt"
        test_content = f"""ArchiveStartTime = {start_time}
ArchiveEndTime = {end_time}
Flight =
Test = Experiment1
Tail = HM
FlightDate = 01/09/25
"""
        test_file.write_text(test_content)
        with pytest.raises(
            ValueError,
            match="Invalid time format in archive info file",
        ):
            IadsUtil.get_irig_times(test_file)
