import FluidAudio
import Foundation

func loadBundledDiarizerModels() async throws -> DiarizerModels {
  let segmentationUrl =
    Bundle.main.url(
      forResource: "pyannote_segmentation",
      withExtension: "mlmodelc",
      subdirectory: "Models"
    )
    ?? Bundle.main.url(forResource: "pyannote_segmentation", withExtension: "mlmodelc")
  let embeddingUrl =
    Bundle.main.url(
      forResource: "wespeaker_v2",
      withExtension: "mlmodelc",
      subdirectory: "Models"
    )
    ?? Bundle.main.url(forResource: "wespeaker_v2", withExtension: "mlmodelc")

  guard let segmentationUrl else {
    throw AudioTranscriptionError.audioFileNotFound(
      name: "pyannote_segmentation",
      ext: "mlmodelc"
    )
  }

  guard let embeddingUrl else {
    throw AudioTranscriptionError.audioFileNotFound(
      name: "wespeaker_v2",
      ext: "mlmodelc"
    )
  }

  return try await DiarizerModels.load(
    localSegmentationModel: segmentationUrl,
    localEmbeddingModel: embeddingUrl
  )
}
