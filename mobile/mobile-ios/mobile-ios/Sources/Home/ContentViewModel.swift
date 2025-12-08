import SwiftUI

@MainActor
final class ContentViewModel: ObservableObject {
  @Published var isTranscribingAudio: Bool = false
  @Published var transcribedText: String = ""
  @Published var isGeneratingJoke: Bool = false
  @Published var jokeText: String = ""
  @Published var isStreamingAudioTranscription: Bool = false
  @Published var streamingTranscribedText: String = ""
  @Published var isPredictingBreastCancer: Bool = false
  @Published var breastCancerPredictionText: String = ""

  var realTimeAudioTranscriptionService: RealTimeAudioTranscriptionService?
}
