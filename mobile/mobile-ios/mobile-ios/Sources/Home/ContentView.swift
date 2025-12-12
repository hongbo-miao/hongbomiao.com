import SwiftUI
@preconcurrency import WhisperKit

struct ContentView: View {
  @StateObject private var contentViewModel: ContentViewModel = ContentViewModel()

  var body: some View {
    ScrollView {
      VStack(spacing: 24) {
        VStack(alignment: .leading, spacing: 4) {
          Text(
            "• Core ML Stable Diffusion (text-to-image generation: SDXL) uses Core ML optimized for both NPU (Apple Neural Engine (ANE)) and GPU (Metal)."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)

          Text(
            "• FluidAudio (voice activity detection (VAD): Silero VAD) uses Core ML optimized exclusively for NPU (Apple Neural Engine (ANE))."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)

          Text(
            "• FluidAudio (speaker diarization: Pyannote Segmentation (segmentation) + WeSpeaker (embedding)) uses Core ML optimized exclusively for NPU (Apple Neural Engine (ANE))."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)

          Text(
            "• WhisperKit (automatic speech recognition (ASR): whisper-tiny.en) uses Core ML optimized for both NPU (Apple Neural Engine (ANE)) and GPU (Metal)."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)

          Text(
            "• MLX Swift LM (large language model (LLM): Qwen3-1.7B-4bit) uses MLX optimized for GPU (Metal)."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)

          Text(
            "• kokoro-ios (text-to-speech (TTS): Kokoro TTS) uses MLX optimized for GPU (Metal)."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)

          Text(
            "• swift-transformers (tokenization)."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)

          Text(
            "• CatBoost (gradient boosting) in Core ML format."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)

          Text(
            "• ModernBERT (masked language modeling (MLM)) in Core ML format."
          )
          .multilineTextAlignment(.leading)
          .fixedSize(horizontal: false, vertical: true)
        }
        .font(.caption)
        .foregroundColor(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)

        Divider()

        VStack(alignment: .leading, spacing: 8) {
          Button(
            action: {
              contentViewModel.handleGenerateStableDiffusionImageButtonTapped()
            },
            label: {
              if contentViewModel.isGeneratingStableDiffusionImage {
                ProgressView()
              } else {
                Text("Generate Image")
              }
            }
          )
          .buttonStyle(.borderedProminent)

          VStack(alignment: .leading, spacing: 4) {
            Text("1. Core ML Stable Diffusion (text-to-image generation: SDXL)")
              .multilineTextAlignment(.leading)
              .fixedSize(horizontal: false, vertical: true)
          }
          .font(.caption)
          .foregroundColor(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
        }

        if let stableDiffusionImage = contentViewModel.stableDiffusionImage {
          VStack(alignment: .leading, spacing: 8) {
            Text("Prompt:")
              .font(.caption)
              .foregroundColor(.secondary)

            Text(AppConfig.stableDiffusionSamplePrompt)
              .frame(maxWidth: .infinity, alignment: .leading)

            Image(decorative: stableDiffusionImage, scale: 1.0)
              .resizable()
              .scaledToFit()
              .cornerRadius(16)
              .overlay(
                RoundedRectangle(cornerRadius: 16)
                  .stroke(.secondary.opacity(0.2))
              )
          }
        }

        Divider()

        VStack(alignment: .leading, spacing: 8) {
          Button(
            action: {
              contentViewModel.handleRealTimeAudioTranscriptionButtonTapped()
            },
            label: {
              if contentViewModel.isStreamingAudioTranscription {
                Text("Stop Live Transcription")
              } else {
                Text("Start Live Transcription")
              }
            }
          )
          .buttonStyle(.borderedProminent)

          VStack(alignment: .leading, spacing: 4) {
            Text(
              "1. FluidAudio (voice activity detector (VAD): Silero VAD)"
            )
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)

            Text(
              "2. WhisperKit (automatic speech recognition (ASR): whisper-tiny.en)"
            )
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)
          }
          .font(.caption)
          .foregroundColor(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
        }

        if !contentViewModel.streamingTranscribedText.isEmpty {
          ScrollView {
            Text(contentViewModel.streamingTranscribedText)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
        }

        Divider()

        VStack(alignment: .leading, spacing: 8) {
          Button(
            action: {
              contentViewModel.handleTranscribeAudioButtonTapped()
            },
            label: {
              if contentViewModel.isTranscribingAudio {
                ProgressView()
              } else {
                Text("Transcribe Audio")
              }
            }
          )
          .buttonStyle(.borderedProminent)

          VStack(alignment: .leading, spacing: 4) {
            Text(
              "1. FluidAudio (voice activity detector (VAD): Silero VAD)"
            )
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)

            Text(
              "2. FluidAudio (speaker diarization: Pyannote Segmentation (segmentation) + WeSpeaker (embedding))"
            )
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)

            Text(
              "3. WhisperKit (automatic speech recognition (ASR): whisper-tiny.en)"
            )
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)
          }
          .font(.caption)
          .foregroundColor(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
        }

        if !contentViewModel.transcribedText.isEmpty {
          ScrollView {
            Text(contentViewModel.transcribedText)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
        }

        Divider()

        VStack(alignment: .leading, spacing: 8) {
          Button(
            action: {
              contentViewModel.handleGenerateJokeButtonTapped()
            },
            label: {
              if contentViewModel.isGeneratingJoke {
                ProgressView()
              } else {
                Text("Generate Joke")
              }
            }
          )
          .buttonStyle(.borderedProminent)

          VStack(alignment: .leading, spacing: 4) {
            Text("1. MLX Swift LM (large language model (LLM): Qwen3-1.7B-4bit)")
              .multilineTextAlignment(.leading)
              .fixedSize(horizontal: false, vertical: true)
            Text("2. kokoro-ios (text-to-speech (TTS): Kokoro TTS)")
              .multilineTextAlignment(.leading)
              .fixedSize(horizontal: false, vertical: true)
          }
          .font(.caption)
          .foregroundColor(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
        }

        if !contentViewModel.jokeText.isEmpty {
          ScrollView {
            Text(contentViewModel.jokeText)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
        }

        Divider()

        VStack(alignment: .leading, spacing: 8) {
          Button(
            action: {
              contentViewModel.handlePredictBreastCancerButtonTapped()
            },
            label: {
              if contentViewModel.isPredictingBreastCancer {
                ProgressView()
              } else {
                Text("Predict Breast Cancer Risk")
              }
            }
          )
          .buttonStyle(.borderedProminent)

          VStack(alignment: .leading, spacing: 4) {
            Text("1. CatBoost (gradient boosting) in Core ML format")
              .multilineTextAlignment(.leading)
              .fixedSize(horizontal: false, vertical: true)
          }
          .font(.caption)
          .foregroundColor(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
        }

        if !contentViewModel.breastCancerPredictionText.isEmpty {
          Text(contentViewModel.breastCancerPredictionText)
            .frame(maxWidth: .infinity, alignment: .leading)
        }

        Divider()

        VStack(alignment: .leading, spacing: 8) {
          Button(
            action: {
              contentViewModel.handleRunModernBertMaskedLanguageModelButtonTapped()
            },
            label: {
              if contentViewModel.isRunningModernBertMaskedLanguageModel {
                ProgressView()
              } else {
                Text("Predict Masked Tokens")
              }
            }
          )
          .buttonStyle(.borderedProminent)

          VStack(alignment: .leading, spacing: 4) {
            Text("1. swift-transformers (tokenizer)")
              .multilineTextAlignment(.leading)
              .fixedSize(horizontal: false, vertical: true)
            Text("2. ModernBERT (masked language modeling (MLM)) in Core ML format")
              .multilineTextAlignment(.leading)
              .fixedSize(horizontal: false, vertical: true)
          }
          .font(.caption)
          .foregroundColor(.secondary)
          .frame(maxWidth: .infinity, alignment: .leading)
        }

        if !contentViewModel.modernBertPredictionDescription.isEmpty {
          VStack(alignment: .leading, spacing: 8) {
            Text("Sentence: The [MASK] of [MASK] is Paris.")
              .frame(maxWidth: .infinity, alignment: .leading)

            ScrollView {
              Text(contentViewModel.modernBertPredictionDescription)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
          }
        }
      }
      .padding()
    }
  }
}
struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
