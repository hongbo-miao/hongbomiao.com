#include <SoapySDR/Constants.h>
#include <SoapySDR/Formats.h>
#include <liquid/liquid.h>

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Errors.hpp>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <print>
#include <stdexcept>
#include <vector>

namespace {

constexpr double WBFM_STATION_HZ = 94900000.0;
constexpr double TUNER_GAIN_DB = 40.0;
constexpr char DEVICE_ARGUMENTS[] = "driver=rtlsdr";

// Audio is written to stdout (for playback). The FM multiplex (for RDS decoding) is a second
// binary stream, so it needs its own destination: file descriptor 3, the first one free after
// stdin (0), stdout (1) and stderr (2). If the caller redirects file descriptor 3 to an RDS
// decoder, the multiplex is written there too; otherwise fdopen below returns null and only audio
// is produced.
constexpr int MPX_SIDE_CHANNEL_FD = 3;

// RTL-SDR Blog V4 runs cleanly up to 2.4 MS/s. A first decimation by 10 brings
// that to a 240 kHz intermediate rate; a second decimation by 5 reaches 48 kHz.
constexpr int DEVICE_SAMPLE_RATE_HZ = 2400000;
constexpr unsigned int CHANNEL_DECIMATION = 10;
constexpr unsigned int AUDIO_DECIMATION = 5;
constexpr int INTERMEDIATE_RATE_HZ = DEVICE_SAMPLE_RATE_HZ / CHANNEL_DECIMATION;
constexpr int AUDIO_SAMPLE_RATE_HZ = INTERMEDIATE_RATE_HZ / AUDIO_DECIMATION;

// Broadcast FM: ~75 kHz peak deviation, 75 us de-emphasis (Americas/Korea), and
// mono audio capped at 15 kHz (the 19 kHz stereo pilot sits just above it).
constexpr double FM_PEAK_DEVIATION_HZ = 75000.0;
constexpr double DEEMPHASIS_SECONDS = 75e-6;
constexpr double AUDIO_CUTOFF_HZ = 15000.0;

// Fixed output volume. freqdem already maps peak deviation to +/-1, so no AGC is
// needed; an AGC would over-react to the demod "clicks" caused by the RTL-SDR DC
// spike sitting on the carrier and pump the program audio down toward silence.
//
// Liquid's firdecim Kaiser prototype has a passband gain equal to its decimation
// factor, so the audio decimator multiplies the signal by AUDIO_DECIMATION. Undo
// that here so the level lands near +/-1 instead of clipping hard.
constexpr float OUTPUT_VOLUME = 0.7f;
constexpr float OUTPUT_SCALE = OUTPUT_VOLUME / static_cast<float>(AUDIO_DECIMATION);

// Filter design quality and one I/Q read worth of input samples.
constexpr unsigned int FILTER_SEMI_LENGTH = 8;
constexpr float STOPBAND_ATTENUATION_DB = 60.0f;
constexpr size_t READ_CHUNK_SAMPLES = CHANNEL_DECIMATION * 8192;

volatile std::sig_atomic_t keep_running = 1;

void handle_stop_signal(int) { keep_running = 0; }

// RAII wrapper for a Liquid-DSP object: holds the opaque handle and calls its
// destroy function on scope exit. Implicitly converts back to the handle so it
// passes straight into the liquid execute functions.
template <typename Handle, int (*Destroy)(Handle)>
class LiquidObject {
 public:
  explicit LiquidObject(Handle handle) : handle_(handle) {
    if (handle_ == nullptr) {
      throw std::runtime_error("Failed to create Liquid-DSP object");
    }
  }
  ~LiquidObject() {
    if (handle_ != nullptr) {
      Destroy(handle_);
    }
  }
  LiquidObject(const LiquidObject&) = delete;
  LiquidObject& operator=(const LiquidObject&) = delete;
  operator Handle() const { return handle_; }

 private:
  Handle handle_;
};

}  // namespace

int main() {
  std::signal(SIGINT, handle_stop_signal);
  std::signal(SIGTERM, handle_stop_signal);
  // If the fd 3 RDS decoder exits first, writing to its closed pipe raises
  // SIGPIPE, which by default kills this process. Ignore it and detect the
  // broken pipe through the failed write instead.
  std::signal(SIGPIPE, SIG_IGN);

  // When fd 3 is wired to redsea (https://github.com/windytan/redsea), emit the
  // demodulated multiplex there so a single run produces both audio on stdout
  // and RDS text. The multiplex still carries the 57 kHz RDS subcarrier. fdopen
  // returns null when fd 3 is not redirected (the plain "play audio only" case).
  FILE* mpx_side_channel = fdopen(MPX_SIDE_CHANNEL_FD, "wb");

  // SoapySDR source: complex float samples from the RTL-SDR (driver=rtlsdr).
  // The C++ API throws on failure, so a try/catch reports any setup error.
  SoapySDR::Device* sdr_source = nullptr;
  SoapySDR::Stream* stream = nullptr;
  try {
    sdr_source = SoapySDR::Device::make(DEVICE_ARGUMENTS);
    sdr_source->setSampleRate(SOAPY_SDR_RX, 0, DEVICE_SAMPLE_RATE_HZ);
    sdr_source->setFrequency(SOAPY_SDR_RX, 0, WBFM_STATION_HZ);
    sdr_source->setGainMode(SOAPY_SDR_RX, 0, false);
    sdr_source->setGain(SOAPY_SDR_RX, 0, TUNER_GAIN_DB);

    // Liquid-DSP chain, one object per stage:
    //   1. firdecim_crcf  - channel select low-pass + decimate x10 (2.4 MS/s -> 240 kHz)
    //   2. freqdem        - FM quadrature demodulation
    //   3. iirfilt_rrrf   - 75 us de-emphasis (single-pole IIR low-pass)
    //   4. firdecim_rrrf  - audio low-pass + decimate x5 (240 kHz -> 48 kHz)
    //   5. iirfilt_rrrf   - audio low-pass at 15 kHz (drops the 19 kHz stereo pilot)
    //   6. iirfilt_rrrf   - DC blocker (high-pass) removing the discriminator's DC term
    LiquidObject<firdecim_crcf, firdecim_crcf_destroy> channel_filter(
      firdecim_crcf_create_kaiser(CHANNEL_DECIMATION, FILTER_SEMI_LENGTH, STOPBAND_ATTENUATION_DB));

    // freqdem's modulation factor maps peak deviation to a full-scale (+/-1) output.
    float modulation_factor = static_cast<float>(FM_PEAK_DEVIATION_HZ / INTERMEDIATE_RATE_HZ);
    LiquidObject<freqdem, freqdem_destroy> fm_demodulator(freqdem_create(modulation_factor));

    // 75 us de-emphasis: y[n] = (1 - a) x[n] + a y[n-1], a = exp(-1 / (tau * Fs)).
    float deemphasis_pole =
      std::exp(-1.0f / static_cast<float>(DEEMPHASIS_SECONDS * INTERMEDIATE_RATE_HZ));
    float deemphasis_feedforward[1] = {1.0f - deemphasis_pole};
    float deemphasis_feedback[2] = {1.0f, -deemphasis_pole};
    LiquidObject<iirfilt_rrrf, iirfilt_rrrf_destroy> de_emphasis(
      iirfilt_rrrf_create(deemphasis_feedforward, 1, deemphasis_feedback, 2));

    LiquidObject<firdecim_rrrf, firdecim_rrrf_destroy> audio_decimator(
      firdecim_rrrf_create_kaiser(AUDIO_DECIMATION, FILTER_SEMI_LENGTH, STOPBAND_ATTENUATION_DB));

    float audio_cutoff_normalized = static_cast<float>(AUDIO_CUTOFF_HZ / AUDIO_SAMPLE_RATE_HZ);
    LiquidObject<iirfilt_rrrf, iirfilt_rrrf_destroy> audio_lowpass(
      iirfilt_rrrf_create_lowpass(5, audio_cutoff_normalized));

    // DC blocker: a frequency offset between the SDR tuning and the carrier turns
    // into a constant (DC) term out of the FM discriminator. A speaker is AC
    // coupled, so that DC is inaudible and would just bias the signal toward the
    // clamp - remove it with a one-pole high-pass: y[n] = x[n] - x[n-1] + R y[n-1].
    float dc_blocker_pole = 0.999f;  // corner ~7.6 Hz at 48 kHz
    float dc_blocker_feedforward[2] = {1.0f, -1.0f};
    float dc_blocker_feedback[2] = {1.0f, -dc_blocker_pole};
    LiquidObject<iirfilt_rrrf, iirfilt_rrrf_destroy> dc_blocker(
      iirfilt_rrrf_create(dc_blocker_feedforward, 2, dc_blocker_feedback, 2));

    stream = sdr_source->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
    sdr_source->activateStream(stream);

    std::println(stderr, "Receiving wbfm at {:.3f} MHz, {} Hz audio", WBFM_STATION_HZ / 1e6,
                 AUDIO_SAMPLE_RATE_HZ);

    // SOAPY_SDR_CF32 is interleaved float [I, Q], the exact layout of
    // liquid_float_complex, so the read buffer feeds the liquid blocks with no
    // conversion. Using liquid's own complex type (rather than
    // std::complex<float>) keeps this independent of header include order.
    std::vector<liquid_float_complex> iq_samples(READ_CHUNK_SAMPLES);
    std::vector<liquid_float_complex> channel_samples(READ_CHUNK_SAMPLES / CHANNEL_DECIMATION);
    std::vector<float> demodulated(READ_CHUNK_SAMPLES / CHANNEL_DECIMATION);
    std::vector<float> audio_samples(READ_CHUNK_SAMPLES / CHANNEL_DECIMATION / AUDIO_DECIMATION);
    std::vector<int16_t> audio_pcm(READ_CHUNK_SAMPLES / CHANNEL_DECIMATION / AUDIO_DECIMATION);
    std::vector<int16_t> mpx_pcm(READ_CHUNK_SAMPLES / CHANNEL_DECIMATION);

    while (keep_running) {
      void* read_buffers[1] = {iq_samples.data()};
      int flags = 0;
      long long time_ns = 0;
      int read_count =
        sdr_source->readStream(stream, read_buffers, READ_CHUNK_SAMPLES, flags, time_ns, 1000000);
      if (read_count < 0) {
        // Timeouts and overflows/underflows are transient; keep going. Anything
        // else (e.g. the dongle was unplugged) is fatal - break so the loop does
        // not spin at 100% CPU retrying a dead device.
        if (read_count == SOAPY_SDR_TIMEOUT || read_count == SOAPY_SDR_OVERFLOW ||
            read_count == SOAPY_SDR_UNDERFLOW) {
          continue;
        }
        std::println(stderr, "readStream failed: {}", SoapySDR::errToStr(read_count));
        break;
      }
      if (read_count == 0) {
        continue;
      }

      // The SDR may return a count that is not a whole multiple of the channel
      // decimation; only process complete decimation groups this iteration.
      unsigned int channel_count = static_cast<unsigned int>(read_count) / CHANNEL_DECIMATION;
      if (channel_count == 0) {
        continue;
      }

      firdecim_crcf_execute_block(channel_filter, iq_samples.data(), channel_count,
                                  channel_samples.data());
      freqdem_demodulate_block(fm_demodulator, channel_samples.data(), channel_count,
                               demodulated.data());

      // De-emphasis (next) is a low-pass that would erase the 57 kHz RDS
      // subcarrier, so tap the multiplex here, before it runs, and send it to
      // the fd 3 side channel when redsea is attached.
      if (mpx_side_channel != nullptr) {
        for (unsigned int i = 0; i < channel_count; i++) {
          float scaled = std::clamp(demodulated[i], -1.0f, 1.0f);
          mpx_pcm[i] = static_cast<int16_t>(std::lrintf(scaled * 32767.0f));
        }
        size_t written =
          std::fwrite(mpx_pcm.data(), sizeof(int16_t), channel_count, mpx_side_channel);
        // The RDS decoder may have exited; stop writing to the broken pipe so the
        // loop does not waste effort on a sink that will never drain again.
        if (written < channel_count || std::fflush(mpx_side_channel) != 0) {
          std::fclose(mpx_side_channel);
          mpx_side_channel = nullptr;
        }
      }

      iirfilt_rrrf_execute_block(de_emphasis, demodulated.data(), channel_count,
                                 demodulated.data());

      unsigned int audio_count = channel_count / AUDIO_DECIMATION;
      if (audio_count == 0) {
        continue;
      }
      firdecim_rrrf_execute_block(audio_decimator, demodulated.data(), audio_count,
                                  audio_samples.data());
      iirfilt_rrrf_execute_block(audio_lowpass, audio_samples.data(), audio_count,
                                 audio_samples.data());
      iirfilt_rrrf_execute_block(dc_blocker, audio_samples.data(), audio_count,
                                 audio_samples.data());

      for (unsigned int i = 0; i < audio_count; i++) {
        float scaled = std::clamp(audio_samples[i] * OUTPUT_SCALE, -1.0f, 1.0f);
        audio_pcm[i] = static_cast<int16_t>(std::lrintf(scaled * 32767.0f));
      }

      std::fwrite(audio_pcm.data(), sizeof(int16_t), audio_count, stdout);
      std::fflush(stdout);
    }
  } catch (const std::exception& error) {
    std::println(stderr, "Receiver error: {}", error.what());
    // Close the side channel so the RDS decoder sees EOF at the point of the
    // error rather than at process teardown.
    if (mpx_side_channel != nullptr) {
      std::fclose(mpx_side_channel);
    }
    // The SoapySDR contract requires the stream to be closed before the device
    // is unmade; skipping it can wedge the USB driver until the dongle is
    // physically re-plugged. Mirror the normal-exit cleanup.
    if (sdr_source != nullptr) {
      if (stream != nullptr) {
        sdr_source->deactivateStream(stream);
        sdr_source->closeStream(stream);
      }
      SoapySDR::Device::unmake(sdr_source);
    }
    return EXIT_FAILURE;
  }

  std::println(stderr, "Stopping receiver");
  if (mpx_side_channel != nullptr) {
    std::fclose(mpx_side_channel);
  }
  sdr_source->deactivateStream(stream);
  sdr_source->closeStream(stream);
  SoapySDR::Device::unmake(sdr_source);

  return EXIT_SUCCESS;
}
