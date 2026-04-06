import logging

from pyflink.common.typeinfo import Types
from pyflink.datastream import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor
from shared.telemetry.models.batch_buffer import BatchBuffer
from shared.telemetry.models.batch_result import BatchResult

logger = logging.getLogger(__name__)


class DetectBatchBoundary(KeyedProcessFunction):
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.buffer_state = None
        self.batch_count_state = None

    def open(self, runtime_context: RuntimeContext) -> None:
        self.buffer_state = runtime_context.get_state(
            ValueStateDescriptor("batch-buffer", Types.PICKLED_BYTE_ARRAY()),
        )
        self.batch_count_state = runtime_context.get_state(
            ValueStateDescriptor("batch-count", Types.LONG()),
        )

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
            result = self._build_batch_result(
                buffer,
                publisher_id,
                batch_count,
                is_partial=False,
            )
            yield result.format_result()
            self.batch_count_state.update(batch_count + 1)
            self.buffer_state.clear()
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

    def _build_batch_result(
        self,
        buffer: BatchBuffer,
        publisher_id: str,
        batch_index: int,
        is_partial: bool,
    ) -> BatchResult:
        return BatchResult(
            publisher_id=publisher_id,
            batch_index=batch_index,
            sample_count=buffer.sample_count,
            temperature_values=list(buffer.temperature_values),
            temperature_average=sum(buffer.temperature_values) / buffer.sample_count,
            first_timestamp_ns=buffer.first_timestamp_ns,
            last_timestamp_ns=buffer.last_timestamp_ns,
            is_partial=is_partial,
        )
