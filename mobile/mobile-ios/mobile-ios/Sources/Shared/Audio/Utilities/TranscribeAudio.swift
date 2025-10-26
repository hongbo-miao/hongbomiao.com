import Foundation
@preconcurrency import WhisperKit

func transcribeAudio(audioResourceName: String, audioResourceExtension: String) async throws
  -> String?
{
  let pipeline: WhisperKit
  do {
    pipeline = try await WhisperKitPipelineProvider.loadWhisperKitPipeline()
  } catch {
    throw AudioTranscriptionError.pipelineLoadFailed(error)
  }

  guard
    let audioPath = Bundle.main.url(
      forResource: audioResourceName, withExtension: audioResourceExtension)
  else {
    throw AudioTranscriptionError.audioFileNotFound(
      name: audioResourceName,
      ext: audioResourceExtension
    )
  }

  do {
    let transcriptionResults = try await pipeline.transcribe(audioPath: audioPath.path)
    let transcriptionText = transcriptionResults.first?.text ?? ""
    let trimmedTranscriptionText = transcriptionText.trimmingCharacters(
      in: .whitespacesAndNewlines
    )
    return trimmedTranscriptionText.isEmpty ? nil : trimmedTranscriptionText
  } catch {
    throw AudioTranscriptionError.transcriptionFailed(error)
  }
}
