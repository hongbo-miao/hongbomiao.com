import AVFoundation
import Foundation
@preconcurrency import WhisperKit

final class RealTimeAudioTranscriptionService: @unchecked Sendable {
  private let audioEngine: AVAudioEngine = AVAudioEngine()
  private var audioFileUrl: URL?
  private var audioFile: AVAudioFile?
  private var processingTask: Task<Void, Never>?
  private var isRunning: Bool = false

  private let transcriptionUpdateHandler: (String) -> Void
  private let errorHandler: (Error) -> Void

  init(
    transcriptionUpdateHandler: @escaping (String) -> Void,
    errorHandler: @escaping (Error) -> Void
  ) {
    self.transcriptionUpdateHandler = transcriptionUpdateHandler
    self.errorHandler = errorHandler
  }

  func start() throws {
    guard !isRunning else {
      return
    }

    let audioSession = AVAudioSession.sharedInstance()
    try audioSession.setCategory(
      .playAndRecord,
      mode: .default,
      options: [.allowBluetoothHFP, .allowBluetoothA2DP, .defaultToSpeaker]
    )
    try audioSession.setActive(true)

    let tempDirectoryUrl = FileManager.default.temporaryDirectory
    let fileUrl = tempDirectoryUrl.appendingPathComponent(
      "live-input-\(UUID().uuidString).caf"
    )

    let inputNode = audioEngine.inputNode
    let inputFormat = inputNode.outputFormat(forBus: 0)

    audioFileUrl = fileUrl
    audioFile = try AVAudioFile(forWriting: fileUrl, settings: inputFormat.settings)

    inputNode.removeTap(onBus: 0)
    inputNode.installTap(
      onBus: 0,
      bufferSize: AVAudioFrameCount(Config.realTimeAudioBufferFrameCount),
      format: inputFormat
    ) { [weak self] buffer, _ in
      guard let self, let audioFile = self.audioFile else {
        return
      }

      do {
        try audioFile.write(from: buffer)
      } catch {
        self.errorHandler(error)
      }
    }

    audioEngine.prepare()
    try audioEngine.start()

    isRunning = true

    if let audioFileUrl {
      processingTask = Task { [weak self] in
        guard let self else {
          return
        }

        await self.runProcessingLoop(audioFileUrl: audioFileUrl)
      }
    }
  }

  func stop() {
    guard isRunning else {
      return
    }

    isRunning = false

    processingTask?.cancel()
    processingTask = nil

    audioEngine.inputNode.removeTap(onBus: 0)
    audioEngine.stop()
    audioEngine.reset()

    audioFile = nil

    if let audioFileUrl {
      try? FileManager.default.removeItem(at: audioFileUrl)
    }

    do {
      try AVAudioSession.sharedInstance().setActive(false)
    } catch {
      errorHandler(error)
    }
  }

  private func runProcessingLoop(audioFileUrl: URL) async {
    let pipeline: WhisperKit
    do {
      pipeline = try await WhisperKitPipelineProvider.loadWhisperKitPipeline()
    } catch {
      errorHandler(AudioTranscriptionError.pipelineLoadFailed(error))
      stop()
      return
    }

    while isRunning {
      if Task.isCancelled {
        break
      }

      do {
        try await Task.sleep(nanoseconds: Config.realTimeAudioProcessingIntervalNanosecondCount)
      } catch {
        break
      }

      await processNewSegments(audioFileUrl: audioFileUrl, pipeline: pipeline)
    }
  }

  private func processNewSegments(
    audioFileUrl: URL,
    pipeline: WhisperKit
  ) async {
    let speechSegments: [SpeechSegment]
    do {
      speechSegments = try await detectVoiceActivityFromAudioFile(
        audioFileUrl: audioFileUrl
      )
    } catch VoiceActivityDetectionError.noSpeechDetected {
      return
    } catch {
      errorHandler(error)
      return
    }

    guard !speechSegments.isEmpty else {
      return
    }

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

        transcribedLines.append(trimmedText)
      } catch {
        errorHandler(AudioTranscriptionError.transcriptionFailed(error))
      }
    }

    guard !transcribedLines.isEmpty else {
      return
    }

    let fullTranscriptionText = transcribedLines.joined(separator: "\n")
    transcriptionUpdateHandler(fullTranscriptionText)
  }
}
