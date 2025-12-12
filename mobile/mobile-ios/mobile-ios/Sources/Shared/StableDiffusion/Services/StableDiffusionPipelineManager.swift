import CoreML
import Foundation
import StableDiffusion
import os

private let logger = Logger(
  subsystem: "com.hongbomiao.mobile-ios",
  category: "StableDiffusionPipelineManager"
)

final class StableDiffusionPipelineManager: @unchecked Sendable {
  static let shared = StableDiffusionPipelineManager()

  private var cachedPipeline: StableDiffusionXLPipeline?
  private var isPipelineLoaded = false
  private let lock = NSLock()

  private init() {}

  // https://huggingface.co/apple/coreml-stable-diffusion-xl-base-ios
  @available(iOS 17.0, macOS 14.0, *)
  func getPipeline(
    resourceDirectoryUrl: URL,
    modelConfiguration: MLModelConfiguration,
    shouldReduceMemoryFootprint: Bool
  ) throws -> StableDiffusionXLPipeline {
    lock.lock()
    defer { lock.unlock() }

    if let existingPipeline = cachedPipeline, isPipelineLoaded {
      logger.info("Reusing cached StableDiffusionXLPipeline")
      return existingPipeline
    }

    logger.info("Creating new StableDiffusionXLPipeline")
    let pipeline = try StableDiffusionXLPipeline(
      resourcesAt: resourceDirectoryUrl,
      configuration: modelConfiguration,
      reduceMemory: shouldReduceMemoryFootprint
    )

    logger.info("Loading XL pipeline resources")
    try pipeline.loadResources()

    cachedPipeline = pipeline
    isPipelineLoaded = true

    return pipeline
  }

  func unloadPipeline() {
    lock.lock()
    defer { lock.unlock() }

    if isPipelineLoaded {
      logger.info("Unloading StableDiffusionXLPipeline")
      cachedPipeline?.unloadResources()
      cachedPipeline = nil
      isPipelineLoaded = false
    }
  }
}
