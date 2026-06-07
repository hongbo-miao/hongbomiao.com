import logging
import time

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TX

logger = logging.getLogger(__name__)

DEVICE_ARGUMENTS = "driver=lime"

# 915 MHz sits in the US 902-928 MHz ISM band (FCC Part 15, unlicensed).
# Transmit only into a cable loopback with an attenuator or a dummy load.
CENTER_FREQUENCY_HZ = 915_000_000
SAMPLE_RATE_HZ = 4_000_000

# Baseband tone offset from the center frequency.
# After the TX to RX loopback the received spectrum should peak at this same offset.
TONE_FREQUENCY_HZ = 100_000

TRANSMIT_GAIN_DB = 15.0
RECEIVE_GAIN_DB = 30.0

BUFFER_SAMPLE_COUNT = 4096
LOOPBACK_DURATION_SECONDS = 0.5


def build_tone(sample_count: int) -> np.ndarray:
    sample_indices = np.arange(sample_count)
    phase = 2.0 * np.pi * TONE_FREQUENCY_HZ * sample_indices / SAMPLE_RATE_HZ
    return np.exp(1j * phase).astype(np.complex64)


def configure_device(device: SoapySDR.Device) -> None:
    for direction in (SOAPY_SDR_TX, SOAPY_SDR_RX):
        device.setSampleRate(direction, 0, SAMPLE_RATE_HZ)
        device.setFrequency(direction, 0, CENTER_FREQUENCY_HZ)
    device.setGain(SOAPY_SDR_TX, 0, TRANSMIT_GAIN_DB)
    device.setGain(SOAPY_SDR_RX, 0, RECEIVE_GAIN_DB)


def find_peak_frequency_hz(received: np.ndarray) -> float:
    spectrum = np.fft.fftshift(np.fft.fft(received))
    frequency_bins_hz = np.fft.fftshift(
        np.fft.fftfreq(received.size, d=1.0 / SAMPLE_RATE_HZ),
    )
    peak_index = int(np.argmax(np.abs(spectrum)))
    return float(frequency_bins_hz[peak_index])


def run_loopback(device: SoapySDR.Device) -> None:
    tone = build_tone(BUFFER_SAMPLE_COUNT)
    received = np.empty(BUFFER_SAMPLE_COUNT, dtype=np.complex64)

    transmit_stream = device.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
    receive_stream = device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    device.activateStream(transmit_stream)
    device.activateStream(receive_stream)

    try:
        deadline = time.monotonic() + LOOPBACK_DURATION_SECONDS
        while time.monotonic() < deadline:
            device.writeStream(transmit_stream, [tone], BUFFER_SAMPLE_COUNT)
            device.readStream(receive_stream, [received], BUFFER_SAMPLE_COUNT)
    finally:
        device.deactivateStream(transmit_stream)
        device.deactivateStream(receive_stream)
        device.closeStream(transmit_stream)
        device.closeStream(receive_stream)

    peak_frequency_hz = find_peak_frequency_hz(received)
    logger.info(
        f"Transmitted tone at {TONE_FREQUENCY_HZ / 1e3:.1f} kHz offset, "
        f"received peak at {peak_frequency_hz / 1e3:.1f} kHz offset.",
    )


def main() -> None:
    logger.info(
        f"Opening LimeSDR Mini 2.0 ({DEVICE_ARGUMENTS}) at "
        f"{CENTER_FREQUENCY_HZ / 1e6:.3f} MHz.",
    )
    try:
        device = SoapySDR.Device(DEVICE_ARGUMENTS)
    except Exception:
        logger.exception(
            "Failed to open the LimeSDR. Check that it is connected and that "
            "the Lime driver is installed (driver=lime).",
        )
        return

    configure_device(device)
    run_loopback(device)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
