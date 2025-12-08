import Foundation
import MLXLMCommon

func generateJoke() async throws -> String? {
  let modelContainer = try await LargeLanguageModelContainer.loadLargeLanguageModelContainer()

  let jokeText = try await modelContainer.perform {
    (context: ModelContext) async throws -> String? in
    let jokeInput = try await context.processor.prepare(
      input: UserInput(
        prompt: .text("Tell a joke."),
        additionalContext: ["enable_thinking": false]
      )
    )

    let generateParameters = GenerateParameters(
      maxTokens: AppConfig.llmMaxTokenCount,
      temperature: AppConfig.llmTemperature
    )
    let jokeResult = try MLXLMCommon.generate(
      input: jokeInput,
      parameters: generateParameters,
      context: context
    ) { (_: [Int]) -> GenerateDisposition in .more }

    let trimmedJokeText = jokeResult.output.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmedJokeText.isEmpty ? nil : trimmedJokeText
  }

  return jokeText
}
