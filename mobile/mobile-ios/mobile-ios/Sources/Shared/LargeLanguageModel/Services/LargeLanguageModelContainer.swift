import Foundation
import MLXLLM
import MLXLMCommon

enum LargeLanguageModelContainer {
  private static let modelContainerTask = Task<ModelContainer, Error> {
    try await LLMModelFactory.shared.loadContainer(
      configuration: AppConfig.largeLanguageModelRegistryConfiguration
    ) { _ in }
  }

  static func loadLargeLanguageModelContainer() async throws -> ModelContainer {
    try await modelContainerTask.value
  }
}
