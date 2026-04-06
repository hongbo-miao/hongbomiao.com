from pydantic import BaseModel, Field


class BatchBuffer(BaseModel):
    temperature_values: list[float] = Field(default_factory=list)
    first_timestamp_ns: int = 0
    last_timestamp_ns: int = 0

    def add_sample(self, temperature_c: float, timestamp_ns: int) -> None:
        if len(self.temperature_values) == 0:
            self.first_timestamp_ns = timestamp_ns
        self.last_timestamp_ns = timestamp_ns
        self.temperature_values.append(temperature_c)

    @property
    def sample_count(self) -> int:
        return len(self.temperature_values)

    def clear(self) -> None:
        self.temperature_values = []
        self.first_timestamp_ns = 0
        self.last_timestamp_ns = 0
