import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pythoncom
import win32com.client

logger = logging.getLogger(__name__)


class IadsUtil:
    @staticmethod
    def show_version_from_file(iads_config: Any, iads_config_path: Path) -> None:  # noqa: ANN401
        try:
            version = iads_config.VersionFromFile(iads_config_path)
            logger.info(f"{version = }")
        except Exception:
            logger.exception(f"{iads_config_path = }")

    @staticmethod
    def execute_query(iads_config: Any, query: str) -> None:  # noqa: ANN401
        try:
            logger.info(f"{query = }")
            results: list[Any] | None = iads_config.Query(query)
            if results:
                for result in results:
                    logger.info(f"{result = }")
        except Exception:
            logger.exception("Failed to process IADS config")

    @staticmethod
    def convert_irig_time_to_year_relative_time_ns(
        irig_time: str,
        year: int,
    ) -> int:
        day, hour, minute, second = irig_time.split(":")
        second, millisecond = second.split(".")

        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=int(day),
            hours=int(hour),
            minutes=int(minute),
            seconds=int(second),
            milliseconds=int(millisecond),
        )
        return int(Decimal(str(dt.timestamp())) * Decimal("1e9"))

    @staticmethod
    def parse_markers(
        results: tuple[str],
        year: int,
    ) -> list[dict[str, str | int]]:
        markers: list[dict[str, str | int]] = []
        for result in results:
            user, time, comment, property_bag = result.replace("\x00", "").split("|")
            markers.append(
                {
                    "time": IadsUtil.convert_irig_time_to_year_relative_time_ns(
                        time,
                        year,
                    ),
                    "comment": comment,
                    "user": user,
                    "property_bag": property_bag,
                },
            )
        return markers

    @staticmethod
    def get_markers(iads_config: Any, year: int) -> list[dict[str, str | int]]:  # noqa: ANN401
        try:
            query = "select User, Time, Comment, PropertyBag from EventMarkerLog"
            results: tuple[str] = iads_config.Query(query)
            return IadsUtil.parse_markers(results, year)
        except Exception:  # noqa: BLE001
            # Return empty list if EventMarkerLog table does not exist
            return []

    @staticmethod
    def parse_test_points(
        results: tuple[str],
        year: int,
    ) -> list[dict[str, str | int]]:
        markers: list[dict[str, str | int]] = []
        for result in results:
            (
                user,
                test_point,
                description,
                maneuver,
                start_time,
                end_time,
                property_bag,
            ) = result.replace("\x00", "").split("|")
            markers.append(
                {
                    "user": user,
                    "test_point": test_point,
                    "description": description,
                    "maneuver": maneuver,
                    "start_time": IadsUtil.convert_irig_time_to_year_relative_time_ns(
                        start_time,
                        year,
                    ),
                    "end_time": IadsUtil.convert_irig_time_to_year_relative_time_ns(
                        end_time,
                        year,
                    ),
                    "property_bag": property_bag,
                },
            )
        return markers

    @staticmethod
    def get_test_points(iads_config: Any, year: int) -> list[dict[str, str | int]]:  # noqa: ANN401
        try:
            query = "select User, TestPoint, Description, Maneuver, StartTime, StopTime, PropertyBag from TestPointLog"
            results: tuple[str] = iads_config.Query(query)
            return IadsUtil.parse_test_points(results, year)
        except Exception:  # noqa: BLE001
            # Return empty list if TestPointLog table does not exist
            return []

    @staticmethod
    def process_config(iads_config_path: Path) -> None:
        iads_config: Any | None = None
        try:
            year = 2025
            pythoncom.CoInitialize()
            iads_config = win32com.client.Dispatch("IadsConfigInterface.IadsConfig")

            IadsUtil.show_version_from_file(iads_config, iads_config_path)
            iads_config.Open(iads_config_path, True)  # noqa: FBT003

            markers = IadsUtil.get_markers(iads_config, year)
            test_points = IadsUtil.get_test_points(iads_config, year)
            logger.info(f"{markers = }")
            logger.info(f"{test_points = }")

            iads_config.Close(True)  # noqa: FBT003
        except Exception:
            logger.exception("Failed to close IADS config")
        finally:
            # Clean up COM resources
            if iads_config:
                iads_config = None
            pythoncom.CoUninitialize()
