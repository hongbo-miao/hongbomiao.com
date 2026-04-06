import logging
from typing import Any

import pulsar
from pyflink.common import Types
from pyflink.datastream import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor
from shared.telemetry.models.batch_buffer import BatchBuffer
from shared.telemetry.models.batch_result_record import BatchResultRecord

logger = logging.getLogger(__name__)


class DetectBatchBoundary(KeyedProcessFunction):
    def __init__(
        self,
        batch_size: int,
        pulsar_service_url: str,
        output_topic: str,
    ) -> None:
        self.batch_size = batch_size
        self.pulsar_service_url = pulsar_service_url
        self.output_topic = output_topic
        self.buffer_state: Any = None
        self.batch_count_state: Any = None
        self.pulsar_client: Any = None
        self.pulsar_producer: Any = None

    def open(self, runtime_context: RuntimeContext) -> None:
        self.buffer_state = runtime_context.get_state(
            ValueStateDescriptor("batch-buffer", Types.PICKLED_BYTE_ARRAY()),
        )
        self.batch_count_state = runtime_context.get_state(
            ValueStateDescriptor("batch-count", Types.LONG()),
        )
        self.pulsar_client = pulsar.Client(self.pulsar_service_url)
        self.pulsar_producer = self.pulsar_client.create_producer(
            self.output_topic,
            schema=pulsar.schema.AvroSchema(BatchResultRecord),
        )

    def close(self) -> None:
        if self.pulsar_producer is not None:
            self.pulsar_producer.close()
        if self.pulsar_client is not None:
            self.pulsar_client.close()

    def process_element(self, value, context: KeyedProcessFunction.Context):  # noqa: ANN001, ANN201, ARG002
        buffer = self._load_buffer()

        publisher_id = value[0]
        timestamp_ns = value[1]
        temperature_c = value[2]
        if temperature_c is None:
            return

        buffer.add_sample(temperature_c, timestamp_ns)

        if buffer.sample_count >= self.batch_size:
            batch_count = self._load_batch_count()
            temperature_average = sum(buffer.temperature_values) / buffer.sample_count

            batch_result = BatchResultRecord(
                publisher_id=publisher_id,
                batch_index=batch_count,
                sample_count=buffer.sample_count,
                temperature_average=temperature_average,
                first_timestamp_ns=buffer.first_timestamp_ns,
                last_timestamp_ns=buffer.last_timestamp_ns,
            )
            self.pulsar_producer.send(batch_result)

            self.batch_count_state.update(batch_count + 1)
            self.buffer_state.clear()

            yield f"Batch {batch_count} for {publisher_id}: sample_count={buffer.sample_count}, temperature_average={temperature_average:.2f}"
            return

        self._save_buffer(buffer)

    def _load_buffer(self) -> BatchBuffer:
        raw = self.buffer_state.value()
        if raw is None:
            return BatchBuffer()
        return BatchBuffer.model_validate_json(raw)

    def _save_buffer(self, buffer: BatchBuffer) -> None:
        self.buffer_state.update(buffer.model_dump_json().encode())

    def _load_batch_count(self) -> int:
        count = self.batch_count_state.value()
        return count if count is not None else 0
