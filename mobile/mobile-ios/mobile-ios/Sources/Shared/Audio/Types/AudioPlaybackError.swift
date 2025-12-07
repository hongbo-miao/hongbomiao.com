import Foundation

enum AudioPlaybackError: Error, LocalizedError {
  case audioFileNotFound(name: String, extension: String)
  case playbackFailed(Error)

  var errorDescription: String? {
    switch self {
    case .audioFileNotFound(let name, let ext):
      "File \(name).\(ext) was not found in the app bundle."
    case .playbackFailed(let underlyingError):
      "Failed to play audio: \(underlyingError.localizedDescription)"
    }
  }
}
