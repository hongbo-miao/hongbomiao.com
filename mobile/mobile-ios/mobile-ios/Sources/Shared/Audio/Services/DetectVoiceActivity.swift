import FluidAudio
import Foundation

struct SpeechSegment {
  let startTime: Double
  let endTime: Double
}

func detectVoiceActivityFromAudioFile(
  audioFileUrl: URL
) async throws -> [SpeechSegment] {
  let vadManager = try await VadManager(
    config: VadConfig(defaultThreshold: AppConfig.sileroVadSpeechThreshold)
  )

  let audioConverter = AudioConverter()
  let samples: [Float]
  do {
    samples = try audioConverter.resampleAudioFile(audioFileUrl)
  } catch {
    throw VoiceActivityDetectionError.audioLoadFailed(error)
  }

  let segmentationConfig = VadSegmentationConfig(
    minSpeechDuration: AppConfig.sileroVadMinSpeechDurationS,
    minSilenceDuration: AppConfig.sileroVadMinSilenceDurationS,
    maxSpeechDuration: AppConfig.sileroVadMaxSegmentDurationS
  )

  let segments = try await vadManager.segmentSpeech(samples, config: segmentationConfig)

  if segments.isEmpty {
    throw VoiceActivityDetectionError.noSpeechDetected
  }

  return segments.map { segment in
    SpeechSegment(startTime: segment.startTime, endTime: segment.endTime)
  }
}
