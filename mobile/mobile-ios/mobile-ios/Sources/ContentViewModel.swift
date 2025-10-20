import SwiftUI

@MainActor
final class ContentViewModel: ObservableObject {
  @Published var isTranscribingAudio: Bool = false
  @Published var transcribedText: String = ""
  @Published var isGeneratingJoke: Bool = false
  @Published var jokeText: String = ""
}
