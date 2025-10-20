import Foundation
import MLXLLM
import MLXLMCommon

enum LargeLanguageModelContainer {
  private static let modelContainerTask = Task<ModelContainer, Error> {
    try await LLMModelFactory.shared.loadContainer(
      // https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/MLXChatExample/Services/MLXService.swift
      configuration: LLMRegistry.qwen3_1_7b_4bit
    ) { _ in }
  }

  static func loadLargeLanguageModelContainer() async throws -> ModelContainer {
    try await modelContainerTask.value
  }
}
