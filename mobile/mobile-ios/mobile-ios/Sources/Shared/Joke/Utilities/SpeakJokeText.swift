import Foundation
import Swift_TTS

func speakJokeText(jokeText: String) async throws {
  let trimmedJokeText = jokeText.trimmingCharacters(in: .whitespacesAndNewlines)
  guard !trimmedJokeText.isEmpty else {
    return
  }

  let sesameSession = try await SesameSession(voice: .conversationalA)
  _ = try await sesameSession.generate(for: trimmedJokeText)
}
