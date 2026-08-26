from pydantic import BaseModel


class TelemetryReading(BaseModel):
    device_id: str
    timestamp_ns: int
    temperature_c: float | None = None
    humidity_pct: float | None = None
