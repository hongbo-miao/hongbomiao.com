import logging
import signal
import sys
from types import FrameType

from gnuradio import analog, audio, blocks, gr, soapy
from gnuradio import filter as gr_filter
from gnuradio.filter import firdes

logger = logging.getLogger(__name__)

MODULATION = "wbfm"
CENTER_FREQUENCY_HZ = 94_900_000
# MODULATION = "am"
# CENTER_FREQUENCY_HZ = 126_950_000

TUNER_GAIN_DB = 40.0
DEVICE_ARGUMENTS = "driver=rtlsdr"

# RTL-SDR Blog V4 runs cleanly up to 2.4 MS/s. A first decimation by 10 brings
# that to a 240 kHz intermediate rate; a second decimation by 5 reaches 48 kHz.
DEVICE_SAMPLE_RATE_HZ = 2_400_000
CHANNEL_DECIMATION = 10
AUDIO_DECIMATION = 5
INTERMEDIATE_RATE_HZ = DEVICE_SAMPLE_RATE_HZ // CHANNEL_DECIMATION
AUDIO_SAMPLE_RATE_HZ = INTERMEDIATE_RATE_HZ // AUDIO_DECIMATION

# A broadcast FM channel is ~200 kHz wide; AM aviation voice is ~8 kHz.
FM_CHANNEL_CUTOFF_HZ = 100_000
FM_CHANNEL_TRANSITION_HZ = 30_000
AM_CHANNEL_CUTOFF_HZ = 5_000
AM_CHANNEL_TRANSITION_HZ = 2_000


class RadioReceiver(gr.top_block):
    def __init__(self) -> None:
        gr.top_block.__init__(self, "Radio receiver")

        # SoapySDR source: complex float samples from the RTL-SDR (driver=rtlsdr).
        self.sdr_source = soapy.source(DEVICE_ARGUMENTS, "fc32", 1, "", "", [""], [""])
        self.sdr_source.set_sample_rate(0, DEVICE_SAMPLE_RATE_HZ)
        self.sdr_source.set_frequency(0, CENTER_FREQUENCY_HZ)
        self.sdr_source.set_gain_mode(0, False)  # noqa: FBT003
        self.sdr_source.set_gain(0, TUNER_GAIN_DB)

        self.audio_sink = audio.sink(AUDIO_SAMPLE_RATE_HZ, "", True)  # noqa: FBT003

        if MODULATION == "wbfm":
            self.connect_wideband_fm()
        elif MODULATION == "am":
            self.connect_am()
        else:
            msg = f"Unsupported MODULATION: {MODULATION}"
            raise ValueError(msg)

    def connect_wideband_fm(self) -> None:
        # Wide channel filter (2.4 MS/s -> 240 kHz), then GNU Radio's WBFM receiver:
        # quadrature demodulation, 75 us de-emphasis, audio low-pass and final decimation to AUDIO_SAMPLE_RATE_HZ.
        channel_taps = firdes.low_pass(
            1.0,
            DEVICE_SAMPLE_RATE_HZ,
            FM_CHANNEL_CUTOFF_HZ,
            FM_CHANNEL_TRANSITION_HZ,
        )
        self.channel_filter = gr_filter.fir_filter_ccf(CHANNEL_DECIMATION, channel_taps)
        self.fm_demodulator = analog.wfm_rcv(INTERMEDIATE_RATE_HZ, AUDIO_DECIMATION)
        self.connect(
            self.sdr_source,
            self.channel_filter,
            self.fm_demodulator,
            self.audio_sink,
        )

    def connect_am(self) -> None:
        # Two-stage filter isolates the narrow AM channel (2.4 MS/s -> 240 kHz -> 48 kHz), then envelope detection recovers the audio:
        # take the magnitude, block the carrier DC, and normalize the level with an AGC.
        wide_taps = firdes.low_pass(
            1.0,
            DEVICE_SAMPLE_RATE_HZ,
            FM_CHANNEL_CUTOFF_HZ,
            FM_CHANNEL_TRANSITION_HZ,
        )
        self.channel_filter = gr_filter.fir_filter_ccf(CHANNEL_DECIMATION, wide_taps)
        narrow_taps = firdes.low_pass(
            1.0,
            INTERMEDIATE_RATE_HZ,
            AM_CHANNEL_CUTOFF_HZ,
            AM_CHANNEL_TRANSITION_HZ,
        )
        self.am_channel_filter = gr_filter.fir_filter_ccf(AUDIO_DECIMATION, narrow_taps)
        self.am_envelope = blocks.complex_to_mag()
        self.dc_blocker = gr_filter.dc_blocker_ff(256, True)  # noqa: FBT003
        self.audio_gain = analog.agc_ff(1e-4, 0.5, 1.0)
        self.connect(
            self.sdr_source,
            self.channel_filter,
            self.am_channel_filter,
            self.am_envelope,
            self.dc_blocker,
            self.audio_gain,
            self.audio_sink,
        )


def main() -> None:
    receiver = RadioReceiver()

    def handle_stop_signal(_signal_number: int, _frame: FrameType | None) -> None:
        logger.info("Stopping receiver")
        receiver.stop()
        receiver.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    frequency_mhz = CENTER_FREQUENCY_HZ / 1e6
    logger.info(
        f"Receiving {MODULATION} at {frequency_mhz:.3f} MHz, {AUDIO_SAMPLE_RATE_HZ} Hz audio.",
    )
    receiver.start()
    receiver.wait()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
