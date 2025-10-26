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
      } catch {
        await MainActor.run {
          isGeneratingJoke = false
        }
      }
    }
  }
}
