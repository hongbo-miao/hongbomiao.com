import logging
import os
from collections.abc import Iterable
from typing import Any

import orjson
import pulsar
from pulsar.schema import AvroSchema
from pyflink.datastream import RuntimeContext
from pyflink.datastream.functions import ProcessWindowFunction
from shared.audio.types.transcript_metric_record import TranscriptMetricRecord

logger = logging.getLogger(__name__)


class ComputeStreamMetrics(ProcessWindowFunction):
    def open(self, _runtime_context: RuntimeContext) -> None:
        pulsar_service_url = os.environ.get(
            "PULSAR_SERVICE_URL",
            "pulsar://pulsar:6650",
        )
        output_topic = os.environ.get(
            "OUTPUT_TOPIC",
            "persistent://public/default/audio-transcript-metrics",
        )
        self._pulsar_client = pulsar.Client(pulsar_service_url)
        self._pulsar_producer = self._pulsar_client.create_producer(
            output_topic,
            schema=AvroSchema(TranscriptMetricRecord),
        )

    def close(self) -> None:
        if hasattr(self, "_pulsar_producer"):
            self._pulsar_producer.close()
        if hasattr(self, "_pulsar_client"):
            self._pulsar_client.close()

    def process(
        self,
        key: str,
        context: ProcessWindowFunction.Context,
        elements: Iterable[Any],
    ) -> Iterable[str]:
        records = sorted(elements, key=lambda row: row[2])
        if not records:
            return

        window = context.window()
        window_size_sec = (window.end - window.start) / 1000.0

        message_count = len(records)
        avg_length_chars = sum(len(row[1]) for row in records) / message_count
        message_rate_per_sec = message_count / window_size_sec

        timestamps_ms = [row[2] / 1_000_000 for row in records]
        avg_gap_ms = None
        min_gap_ms = None
        max_gap_ms = None
        if message_count >= 2:
            gaps = [
                timestamps_ms[index] - timestamps_ms[index - 1]
                for index in range(1, message_count)
            ]
            avg_gap_ms = sum(gaps) / len(gaps)
            min_gap_ms = min(gaps)
            max_gap_ms = max(gaps)

        metric = TranscriptMetricRecord(
            device_id=key,
            window_start_ms=window.start,
            window_end_ms=window.end,
            message_rate_per_sec=round(message_rate_per_sec, 4),
            message_count=message_count,
            avg_length_chars=round(avg_length_chars, 2),
            avg_gap_ms=round(avg_gap_ms, 2) if avg_gap_ms is not None else None,
            min_gap_ms=round(min_gap_ms, 2) if min_gap_ms is not None else None,
            max_gap_ms=round(max_gap_ms, 2) if max_gap_ms is not None else None,
        )

        self._pulsar_producer.send_async(
            metric,
            callback=lambda result, message_id: (
                logger.error(
                    f"Failed to send metric to Pulsar (message_id={message_id}): {result}",
                )
                if result != pulsar.Result.Ok
                else None
            ),
        )
        yield orjson.dumps(
            {
                "device_id": metric.device_id,
                "window_start_ms": metric.window_start_ms,
                "window_end_ms": metric.window_end_ms,
                "message_rate_per_sec": metric.message_rate_per_sec,
                "message_count": metric.message_count,
                "avg_length_chars": metric.avg_length_chars,
                "avg_gap_ms": metric.avg_gap_ms,
                "min_gap_ms": metric.min_gap_ms,
                "max_gap_ms": metric.max_gap_ms,
            },
        ).decode()
