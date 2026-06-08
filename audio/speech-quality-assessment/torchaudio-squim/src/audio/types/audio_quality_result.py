from pydantic import BaseModel


class AudioQualityResult(BaseModel):
    name: str
    duration_seconds: float
    silence_gap_snr_db: float
    speech_fraction: float
    estimated_pesq: float | None = None
    estimated_stoi: float | None = None
    estimated_si_sdr_db: float | None = None
