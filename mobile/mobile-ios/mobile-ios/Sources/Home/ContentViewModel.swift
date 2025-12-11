import CoreGraphics
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
  @Published var isRunningModernBertMaskedLanguageModel: Bool = false
  @Published var modernBertPredictionDescription: String = ""
  @Published var isGeneratingStableDiffusionImage: Bool = false
  @Published var stableDiffusionImage: CGImage?

  var realTimeAudioTranscriptionService: RealTimeAudioTranscriptionService?
}
