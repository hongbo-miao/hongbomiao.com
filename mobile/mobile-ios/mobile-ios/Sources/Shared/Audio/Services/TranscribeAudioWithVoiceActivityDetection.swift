import Foundation
@preconcurrency import WhisperKit

func transcribeAudioWithVoiceActivityDetection(
  audioResourceName: String,
  audioResourceExtension: String
) async throws -> String? {
  // Step 1: Get audio file path
  guard
    let audioFileUrl = Bundle.main.url(
      forResource: audioResourceName,
      withExtension: audioResourceExtension
    )
  else {
    throw AudioTranscriptionError.audioFileNotFound(
      name: audioResourceName,
      ext: audioResourceExtension
    )
  }

  // Step 2: Detect voice activity and get speech segments
  let speechSegments = try await detectVoiceActivityFromAudioFile(
    audioFileUrl: audioFileUrl
  )

  // Log detected speech segments
  for (index, segment) in speechSegments.enumerated() {
    print(
      String(
        format: "Speech segment %d: %.2f - %.2f seconds",
        index + 1,
        segment.startTime,
        segment.endTime
      )
    )
  }

  // Step 3: Load WhisperKit pipeline
  let pipeline: WhisperKit
  do {
    pipeline = try await WhisperKitPipelineProvider.loadWhisperKitPipeline()
  } catch {
    throw AudioTranscriptionError.pipelineLoadFailed(error)
  }

  // Step 4: Transcribe each speech segment separately
  var transcribedLines: [String] = []

  for segment in speechSegments {
    do {
      let decodingOptions = DecodingOptions(
        clipTimestamps: [Float(segment.startTime), Float(segment.endTime)]
      )
      let transcriptionResults = try await pipeline.transcribe(
        audioPath: audioFileUrl.path,
        decodeOptions: decodingOptions
      )
      let transcriptionText = transcriptionResults.first?.text ?? ""
      let trimmedText = transcriptionText.trimmingCharacters(
        in: CharacterSet.whitespacesAndNewlines
      )
      if !trimmedText.isEmpty {
        let formattedLine = String(
          format: "[%.1f-%.1fs] %@",
          segment.startTime,
          segment.endTime,
          trimmedText
        )
        transcribedLines.append(formattedLine)
      }
    } catch {
      throw AudioTranscriptionError.transcriptionFailed(error)
    }
  }

  return transcribedLines.isEmpty ? nil : transcribedLines.joined(separator: "\n")
}
