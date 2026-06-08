# Frame length for the silence-gap SNR estimate; 20 ms is the usual short-time analysis window for speech.
FRAME_DURATION_SECONDS = 0.02

# Frames at or below this energy percentile are treated as the noise floor; frames at or above the speech percentile are treated as voice.
QUIET_PERCENTILE = 20.0
SPEECH_PERCENTILE = 80.0

# TorchAudio SQUIM operates at 16 kHz.
# The audio is scored in fixed windows and averaged, because SQUIM cost scales with clip length and PESQ is defined on short utterances.
SQUIM_SAMPLE_RATE_HZ = 16_000
SQUIM_WINDOW_SECONDS = 10.0
