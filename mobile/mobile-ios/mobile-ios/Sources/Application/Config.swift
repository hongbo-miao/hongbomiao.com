import Foundation

enum Config {
  static let llmMaxTokenCount: Int = 1024
  static let llmTemperature: Float = 0.6
  static let audioResourceName: String = "audio"
  static let audioResourceExtension: String = "wav"
  static let kokoroVoiceStyleName: String = "af_bella"
  static let mlxGpuMemoryLimitByteCount: Int = 500 * 1024 * 1024  // 500 MiB
}
