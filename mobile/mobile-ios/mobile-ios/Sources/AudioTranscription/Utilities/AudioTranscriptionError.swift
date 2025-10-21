import Foundation

enum AudioTranscriptionError: Error, LocalizedError {
  case pipelineLoadFailed(Error)
  case audioFileNotFound(name: String, ext: String)
  case transcriptionFailed(Error)

  var errorDescription: String? {
    switch self {
    case .pipelineLoadFailed(let underlyingError):
      "Failed to load WhisperKit pipeline: \(underlyingError.localizedDescription)"
    case .audioFileNotFound(let name, let ext):
      "File \(name).\(ext) was not found in the app bundle."
    case .transcriptionFailed(let underlyingError):
      "Failed to transcribe audio: \(underlyingError.localizedDescription)"
    }
  }
}
