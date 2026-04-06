from pydantic import BaseModel


class BatchResult(BaseModel):
    publisher_id: str
    batch_index: int
    sample_count: int
    temperature_values: list[float]
    temperature_average: float
    first_timestamp_ns: int
    last_timestamp_ns: int
    is_partial: bool

    def format_result(self) -> str:
        prefix = "[PARTIAL] " if self.is_partial else ""
        duration_ms = (self.last_timestamp_ns - self.first_timestamp_ns) / 1_000_000
        temperature_values_string = ", ".join(
            f"{value:.1f}" for value in self.temperature_values
        )
        return (
            f"{prefix}Publisher {self.publisher_id}: "
            f"Batch #{self.batch_index} "
            f"temperature_average={self.temperature_average:.1f} "
            f"sample_count={self.sample_count} "
            f"duration_ms={duration_ms:.0f} "
            f"temperature=[{temperature_values_string}]"
        )
