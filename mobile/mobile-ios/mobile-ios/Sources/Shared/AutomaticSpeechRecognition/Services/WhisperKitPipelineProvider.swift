import Foundation
@preconcurrency import WhisperKit

enum WhisperKitPipelineProvider {
  static let pipelineTask = Task<WhisperKit, Error> {
    try await WhisperKit(
      WhisperKitConfig(
        model: AppConfig.whisperKitModelName,
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
