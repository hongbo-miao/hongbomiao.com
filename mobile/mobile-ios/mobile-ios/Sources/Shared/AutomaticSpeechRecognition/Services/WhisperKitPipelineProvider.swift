import Foundation
@preconcurrency import WhisperKit

enum WhisperKitPipelineProvider {
  static let pipelineTask = Task<WhisperKit, Error> {
    // https://huggingface.co/argmaxinc/whisperkit-coreml/tree/main
    try await WhisperKit(WhisperKitConfig(model: "openai_whisper-tiny.en"))
  }

  static func loadWhisperKitPipeline() async throws -> WhisperKit {
    try await pipelineTask.value
  }
}
