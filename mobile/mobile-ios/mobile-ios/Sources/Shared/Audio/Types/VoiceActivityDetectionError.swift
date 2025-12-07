import Foundation

enum VoiceActivityDetectionError: Error, LocalizedError {
  case audioFileNotFound(name: String, extension: String)
  case audioLoadFailed(Error)
  case noSpeechDetected

  var errorDescription: String? {
    switch self {
    case .audioFileNotFound(let name, let ext):
      "File \(name).\(ext) was not found in the app bundle."
    case .audioLoadFailed(let underlyingError):
      "Failed to load audio file: \(underlyingError.localizedDescription)"
    case .noSpeechDetected:
      "No speech detected in the audio file."
    }
  }
}
