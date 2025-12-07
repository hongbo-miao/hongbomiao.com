import Foundation
@preconcurrency import WhisperKit

enum WhisperKitPipelineProvider {
  static let pipelineTask = Task<WhisperKit, Error> {
    try await WhisperKit(
      WhisperKitConfig(
        // https://huggingface.co/argmaxinc/whisperkit-coreml/tree/main
        model: "openai_whisper-tiny.en",
        computeOptions: ModelComputeOptions(
          audioEncoderCompute: .cpuAndNeuralEngine,
          textDecoderCompute: .cpuAndNeuralEngine,
        ),
      ),
    )
  }

  static func loadWhisperKitPipeline() async throws -> WhisperKit {
    try await pipelineTask.value
  }
}
