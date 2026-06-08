# torchaudio-squim

## Introduction

```shell
file          dur (s)  SNR (dB)  speech %  est PESQ  est STOI  est SI-SDR
-------------------------------------------------------------------------
a               120.0      10.7     20.0%      1.12      0.81         4.0
b               120.0      12.9     20.0%      1.13      0.83         6.4
```

- **Silence-gap SNR (dB), higher is better.** The ratio between the power of the loud (speech) frames and the power of the quiet (noise-floor) frames, found from the natural pauses in speech. It is computed with numpy alone, is deterministic, and needs no reference, so it is always available. Higher means the speech sits further above the background noise.
- **Estimated PESQ (about 1.0 to 4.5), higher is better.** Wideband Perceptual Evaluation of Speech Quality, predicted with no reference by TorchAudio's [SQUIM objective](https://docs.pytorch.org/audio/stable/tutorials/squim_tutorial.html) model. It approximates the mean opinion score a listener would give for overall speech quality, where higher is cleaner.
- **Estimated STOI (0 to 1), higher is better.** Short-Time Objective Intelligibility, predicted by the same SQUIM model. It estimates how intelligible the speech is, where 1 is perfectly intelligible.
- **Estimated SI-SDR (dB), higher is better.** Scale-Invariant Signal-to-Distortion Ratio, predicted by the same SQUIM model. It estimates how much of the signal is the clean speech versus distortion, where higher means less distortion.

SQUIM was trained on wideband speech, so band-limited audio (for example telephone-band, roughly 300-3400 Hz) reads low on absolute PESQ for everyone.
Use PESQ, STOI, and SI-SDR as a relative comparison between the recordings rather than as absolute scores; the silence-gap SNR is the most directly interpretable number.

The silence-gap SNR is meaningful only when a recording actually contains speech with pauses.
A continuous tone, or silence, scores near 0 dB because there is no contrast between speech and quiet.

## Why not SINAD

Classical SINAD is defined on a single known tone, and speech is not a tone, so tone-SINAD does not apply to a voice recording.
The silence-gap SNR and the perceptual estimates above are the speech-appropriate alternatives.
