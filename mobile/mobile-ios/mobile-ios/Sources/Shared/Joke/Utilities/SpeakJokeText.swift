import Foundation
import KokoroSwift
import MLX
import os

private let logger = Logger(subsystem: "com.hongbomiao.mobile-ios", category: "SpeakJokeText")

@MainActor
func speakJokeText(jokeText: String, voiceStyleName: String) async throws {
  let trimmedJokeText = jokeText.trimmingCharacters(in: .whitespacesAndNewlines)
  guard !trimmedJokeText.isEmpty else {
    logger.info("Empty joke text, skipping TTS")
    return
  }

  // Set GPU memory limit to prevent crashes on devices with limited memory
  logger.info("Set GPU memory limit")
  MLX.GPU.set(memoryLimit: AppConfig.mlxGpuMemoryLimitByteCount)

  // Initialize Kokoro TTS
  let modelPath = try KokoroVoiceLoader.getModelPath()
  logger.info("Initializing KokoroTTS with model path: \(modelPath.path)")
  let tts = KokoroTTS(modelPath: modelPath, g2p: .misaki)

  // Load voice style
  logger.info("Loading voice style")
  let voiceEmbedding = try KokoroVoiceLoader.loadVoiceStyle(voiceStyleName: voiceStyleName)

  // Generate audio
  logger.info("Generating audio...")
  let (audioSamples, _) = try tts.generateAudio(
    voice: voiceEmbedding,
    language: .enUS,
    text: trimmedJokeText,
    speed: 1.0
  )
  logger.info("Generated \(audioSamples.count) audio samples")

  // Play the generated audio
  let audioPlayer = AudioPlayer()
  logger.info("Starting audio playback with sample rate: \(KokoroTTS.Constants.samplingRate)")
  await audioPlayer.playAudio(
    samples: audioSamples,
    sampleRate: Double(KokoroTTS.Constants.samplingRate)
  )
  logger.info("Audio playback completed")
}
