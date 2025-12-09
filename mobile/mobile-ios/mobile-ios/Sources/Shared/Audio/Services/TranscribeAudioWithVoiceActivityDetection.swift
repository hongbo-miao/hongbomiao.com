import FluidAudio
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

  // Step 3: Prepare speaker diarization
  let diarizerConfig = DiarizerConfig(
    clusteringThreshold: AppConfig.diarizerClusteringThreshold,
    minSpeechDuration: AppConfig.diarizerMinSpeechDurationS,
    minEmbeddingUpdateDuration: AppConfig.diarizerMinEmbeddingUpdateDurationS,
    minSilenceGap: AppConfig.diarizerMinSilenceGapS,
    numClusters: AppConfig.diarizerExpectedSpeakerCount,
    minActiveFramesCount: AppConfig.diarizerMinActiveFramesCount,
    chunkDuration: AppConfig.diarizerChunkDurationS,
    chunkOverlap: AppConfig.diarizerChunkOverlapS
  )
  let diarizerManager = DiarizerManager(config: diarizerConfig)
  let diarizationSegments: [TimedSpeakerSegment]

  do {
    let diarizerModels = try await loadBundledDiarizerModels()
    diarizerManager.initialize(models: diarizerModels)

    let audioConverter = AudioConverter()
    let diarizationSamples = try audioConverter.resampleAudioFile(audioFileUrl)
    let diarizationResult = try diarizerManager.performCompleteDiarization(
      diarizationSamples
    )
    diarizationSegments = diarizationResult.segments
  } catch DiarizerError.notInitialized {
    throw AudioTranscriptionError.diarizationModelLoadFailed(DiarizerError.notInitialized)
  } catch {
    if error is DiarizerError {
      throw AudioTranscriptionError.diarizationFailed(error)
    }

    throw AudioTranscriptionError.diarizationModelLoadFailed(error)
  }

  defer {
    diarizerManager.cleanup()
  }

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
      if trimmedText.isEmpty {
        continue
      }

      let speakerMatch =
        diarizationSegments
        .map { speakerSegment in
          (
            segment: speakerSegment,
            overlap: max(
              0,
              min(Double(speakerSegment.endTimeSeconds), segment.endTime)
                - max(Double(speakerSegment.startTimeSeconds), segment.startTime)
            )
          )
        }
        .max(by: { first, second in first.overlap < second.overlap })

      let speakerLabel: String
      if let match = speakerMatch, match.overlap > 0 {
        speakerLabel = "Speark \(match.segment.speakerId)"
      } else {
        speakerLabel = "Unknown speaker"
      }

      let timeRangeDescription = String(
        format: "%.1f-%.1fs",
        segment.startTime,
        segment.endTime
      )

      let formattedLine = "[\(speakerLabel), \(timeRangeDescription)] \(trimmedText)"
      transcribedLines.append(formattedLine)
    } catch {
      throw AudioTranscriptionError.transcriptionFailed(error)
    }
  }

  return transcribedLines.isEmpty ? nil : transcribedLines.joined(separator: "\n")
}
