import logging

import asyncpg
import pytest
from tenacity import retry, stop_after_delay, wait_fixed

logger = logging.getLogger(__name__)


class TestTelemetryToPostgres:
    @pytest.mark.asyncio
    async def test_telemetry_written_to_postgres(
        self,
        postgres_url: str,
    ) -> None:
        @retry(
            stop=stop_after_delay(120),
            wait=wait_fixed(5.0),
            reraise=True,
        )
        async def _assert_telemetry_in_db(postgres_url: str) -> None:
            connection = await asyncpg.connect(postgres_url)
            try:
                telemetry_count = await connection.fetchval(
                    "select count(*) from telemetry",
                )
                assert telemetry_count >= 3, (
                    f"Expected at least 3 telemetry records, got {telemetry_count}"
                )

                publisher_count = await connection.fetchval(
                    "select count(distinct publisher_id) from telemetry",
                )
                assert publisher_count >= 2, (
                    f"Expected at least 2 distinct publishers, got {publisher_count}"
                )
            finally:
                await connection.close()

        await _assert_telemetry_in_db(postgres_url)

    @pytest.mark.asyncio
    async def test_telemetry_columns_not_null(
        self,
        postgres_url: str,
    ) -> None:
        @retry(
            stop=stop_after_delay(120),
            wait=wait_fixed(5.0),
            reraise=True,
        )
        async def _assert_columns_not_null(postgres_url: str) -> None:
            connection = await asyncpg.connect(postgres_url)
            try:
                row = await connection.fetchrow(
                    "select * from telemetry order by timestamp_ns desc limit 1",
                )
                assert row is not None, "No telemetry rows found"
                assert row["publisher_id"] is not None, "publisher_id is null"
                assert row["timestamp_ns"] is not None, "timestamp_ns is null"
                assert row["temperature_c"] is not None, "temperature_c is null"
                assert row["humidity_pct"] is not None, "humidity_pct is null"
            finally:
                await connection.close()

        await _assert_columns_not_null(postgres_url)
