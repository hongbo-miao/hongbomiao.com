import SwiftUI

extension ContentViewModel {
  func handleGenerateJokeButtonTapped() {
    guard !isGeneratingJoke else {
      return
    }

    isGeneratingJoke = true
    jokeText = ""

    Task {
      do {
        let jokeTextResult = try await generateJoke()
        await MainActor.run {
          jokeText = jokeTextResult ?? ""
          isGeneratingJoke = false
        }

        if let jokeTextResult = jokeTextResult {
          try await speakJokeText(
            jokeText: jokeTextResult, voiceStyleName: AppConfig.kokoroVoiceStyleName)
        }
      } catch {
        await MainActor.run {
          isGeneratingJoke = false
        }
      }
    }
  }
}
