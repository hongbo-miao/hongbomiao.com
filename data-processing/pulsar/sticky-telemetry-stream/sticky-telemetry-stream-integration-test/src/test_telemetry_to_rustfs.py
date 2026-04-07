import logging

import pytest
from botocore.client import BaseClient
from tenacity import retry, stop_after_delay, wait_fixed

logger = logging.getLogger(__name__)


class TestTelemetryToRustfs:
    @pytest.mark.asyncio
    async def test_parquet_files_written_to_rustfs(
        self,
        s3_client: BaseClient,
    ) -> None:
        @retry(
            stop=stop_after_delay(120),
            wait=wait_fixed(5.0),
            reraise=True,
        )
        def _assert_parquet_files_in_rustfs(s3_client: BaseClient) -> None:
            response = s3_client.list_objects_v2(Bucket="telemetry-bucket")
            assert response.get("KeyCount", 0) >= 1, (
                "Expected at least 1 object in telemetry-bucket"
            )
            keys = [obj["Key"] for obj in response.get("Contents", [])]
            parquet_keys = [key for key in keys if key.endswith(".parquet")]
            assert len(parquet_keys) >= 1, (
                f"Expected at least 1 parquet file, got keys: {keys}"
            )

        _assert_parquet_files_in_rustfs(s3_client)
