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
        let transcriptionText = try await transcribeAudioWithVoiceActivityDetection(
          audioResourceName: AppConfig.audioResourceName,
          audioResourceExtension: AppConfig.audioResourceExtension
        )

        await MainActor.run {
          transcribedText = transcriptionText ?? ""
          isTranscribingAudio = false

          do {
            try playAudio(
              audioResourceName: AppConfig.audioResourceName,
              audioResourceExtension: AppConfig.audioResourceExtension
            )
          } catch {
            print("Failed to play audio: \(error.localizedDescription)")
          }
        }
      } catch {
        await MainActor.run {
          isTranscribingAudio = false
        }
      }
    }
  }
}
