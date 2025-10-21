import SwiftUI

extension ContentViewModel {
  func handleTranscribeAudioButtonTapped() {
    guard !isTranscribingAudio else {
      return
    }

    isTranscribingAudio = true
    transcribedText = ""

    Task {
      do {
        let transcriptionText = try await transcribeAudio(
          audioResourceName: Config.audioResourceName,
          audioResourceExtension: Config.audioResourceExtension
        )

        await MainActor.run {
          transcribedText = transcriptionText ?? ""
          isTranscribingAudio = false
        }
      } catch {
        await MainActor.run {
          isTranscribingAudio = false
        }
      }
    }
  }
}
