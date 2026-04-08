import logging

import asyncpg
import pytest
from tenacity import retry, stop_after_delay, wait_fixed

logger = logging.getLogger(__name__)


class TestFlinkBatchResultsToPostgres:
    @pytest.mark.asyncio
    async def test_batch_results_written_to_postgres(
        self,
        postgres_url: str,
    ) -> None:
        @retry(
            stop=stop_after_delay(300),
            wait=wait_fixed(5.0),
            reraise=True,
        )
        async def _assert_batch_results_in_db(postgres_url: str) -> None:
            connection = await asyncpg.connect(postgres_url)
            try:
                batch_result_count = await connection.fetchval(
                    "select count(*) from batch_results",
                )
                assert batch_result_count >= 1, (
                    f"Expected at least 1 batch result, got {batch_result_count}"
                )
            finally:
                await connection.close()

        await _assert_batch_results_in_db(postgres_url)

    @pytest.mark.asyncio
    async def test_batch_results_columns_valid(
        self,
        postgres_url: str,
    ) -> None:
        @retry(
            stop=stop_after_delay(300),
            wait=wait_fixed(5.0),
            reraise=True,
        )
        async def _assert_batch_results_valid(postgres_url: str) -> None:
            connection = await asyncpg.connect(postgres_url)
            try:
                row = await connection.fetchrow(
                    "select * from batch_results order by batch_index desc limit 1",
                )
                assert row is not None, "No batch result rows found"
                assert row["publisher_id"] is not None, "publisher_id is null"
                assert row["sample_count"] >= 10, (
                    f"Expected sample_count >= 10 (batch size), got {row['sample_count']}"
                )
                assert row["temperature_average"] is not None, (
                    "temperature_average is null"
                )
                assert row["first_timestamp_ns"] < row["last_timestamp_ns"], (
                    "first_timestamp_ns should be less than last_timestamp_ns"
                )
            finally:
                await connection.close()

        await _assert_batch_results_valid(postgres_url)
