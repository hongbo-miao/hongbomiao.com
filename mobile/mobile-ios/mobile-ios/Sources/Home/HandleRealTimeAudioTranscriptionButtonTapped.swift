import SwiftUI

extension ContentViewModel {
  func handleRealTimeAudioTranscriptionButtonTapped() {
    if isStreamingAudioTranscription {
      isStreamingAudioTranscription = false
      realTimeAudioTranscriptionService?.stop()
      realTimeAudioTranscriptionService = nil
      return
    }

    isStreamingAudioTranscription = true
    streamingTranscribedText = ""

    let service = RealTimeAudioTranscriptionService(
      transcriptionUpdateHandler: { [weak self] newText in
        Task { @MainActor in
          guard let strongSelf = self else {
            return
          }

          strongSelf.streamingTranscribedText = newText
        }
      },
      errorHandler: { [weak self] error in
        print("Real-time transcription error: \(error.localizedDescription)")
        Task { @MainActor in
          guard let strongSelf = self else {
            return
          }

          strongSelf.isStreamingAudioTranscription = false
        }
      }
    )

    realTimeAudioTranscriptionService = service

    do {
      try service.start()
    } catch {
      isStreamingAudioTranscription = false
    }
  }
}
