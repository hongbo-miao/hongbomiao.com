import Foundation

func transcribeAudio(audioResourceName: String, audioResourceExtension: String) async -> (
  transcribedText: String?, statusMessage: String
) {
  do {
    let pipeline = try await WhisperKitPipelineProvider.loadWhisperKitPipeline()

    guard
      let audioPath = Bundle.main.url(
        forResource: audioResourceName, withExtension: audioResourceExtension)
    else {
      return (
        nil, "File \(audioResourceName).\(audioResourceExtension) was not found in the app bundle."
      )
    }

    let transcriptionResults = try await pipeline.transcribe(audioPath: audioPath.path)

    guard let text = transcriptionResults.first?.text, !text.isEmpty else {
      return (nil, "WhisperKit returned an empty transcription.")
    }
    return (text, "Transcription succeeded.")
  } catch {
    return (nil, "Failed to transcribe audio: \(error.localizedDescription)")
  }
}
