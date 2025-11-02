import Foundation
import KokoroSwift
import MLX
import MLXUtilsLibrary
import os

enum KokoroVoiceLoaderError: Error {
  case modelNotFound
  case voiceStyleNotFound
  case invalidVoiceData
}

@MainActor
final class KokoroVoiceLoader {
  private static let logger = Logger(
    subsystem: "com.hongbomiao.mobile-ios", category: "KokoroVoiceLoader")
  private static let shared = KokoroVoiceLoader()

  private var voices: [String: MLXArray] = [:]
  private var isLoaded = false

  private init() {}

  /// Loads the Kokoro model file from the app bundle
  nonisolated static func getModelPath() throws -> URL {
    guard let modelPath = Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors")
    else {
      throw KokoroVoiceLoaderError.modelNotFound
    }

    guard FileManager.default.fileExists(atPath: modelPath.path) else {
      throw KokoroVoiceLoaderError.modelNotFound
    }

    return modelPath
  }

  /// Loads all voices from the voices.npz file
  private func loadVoices() throws {
    guard !isLoaded else { return }

    guard let voiceFilePath = Bundle.main.url(forResource: "voices", withExtension: "npz") else {
      Self.logger.error("voices.npz file not found in bundle")
      throw KokoroVoiceLoaderError.voiceStyleNotFound
    }

    Self.logger.info("Loading voices from: \(voiceFilePath.path)")

    guard let loadedVoices = NpyzReader.read(fileFromPath: voiceFilePath) else {
      Self.logger.error("Failed to read voices.npz file")
      throw KokoroVoiceLoaderError.invalidVoiceData
    }

    Self.logger.info("Loaded \(loadedVoices.count) voice entries")
    Self.logger.info("Available voice keys: \(loadedVoices.keys.joined(separator: ", "))")

    voices = loadedVoices
    isLoaded = true
  }

  static func loadVoiceStyle(voiceStyleName: String) throws -> MLXArray {
    try shared.loadVoiceStyleInternal(voiceStyleName: voiceStyleName)
  }

  private func loadVoiceStyleInternal(voiceStyleName: String) throws -> MLXArray {
    try loadVoices()

    let voiceKey = "\(voiceStyleName).npy"
    Self.logger.info("Looking for voice key: \(voiceKey)")

    guard let voiceEmbedding = voices[voiceKey] else {
      Self.logger.error(
        "Voice key '\(voiceKey)' not found. Available keys: \(self.voices.keys.joined(separator: ", "))"
      )
      throw KokoroVoiceLoaderError.voiceStyleNotFound
    }

    Self.logger.info("Successfully loaded voice: \(voiceKey)")
    return voiceEmbedding
  }

  /// Returns list of available voice names
  static func getAvailableVoices() throws -> [String] {
    try shared.getAvailableVoicesInternal()
  }

  private func getAvailableVoicesInternal() throws -> [String] {
    try loadVoices()
    return voices.keys.map { String($0.split(separator: ".")[0]) }.sorted()
  }
}
