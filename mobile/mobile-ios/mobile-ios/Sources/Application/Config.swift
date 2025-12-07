import Foundation

enum Config {
  static let llmMaxTokenCount: Int = 1024
  static let llmTemperature: Float = 0.6
  static let audioResourceName: String = "audio"
  static let audioResourceExtension: String = "wav"
  static let kokoroVoiceStyleName: String = "af_bella"
  static let mlxGpuMemoryLimitByteCount: Int = 500 * 1024 * 1024  // 500 MiB
  static let sileroVadSpeechThreshold: Float = 0.5
  static let sileroVadMinSpeechDurationS: Double = 0.25
  static let sileroVadMinSilenceDurationS: Double = 1.0
  static let sileroVadMaxSegmentDurationS: Double = 10.0
  static let realTimeAudioBufferFrameCount: Int = 4096
  static let realTimeAudioProcessingIntervalNanosecondCount: UInt64 = 2 * 1_000_000_000
}
