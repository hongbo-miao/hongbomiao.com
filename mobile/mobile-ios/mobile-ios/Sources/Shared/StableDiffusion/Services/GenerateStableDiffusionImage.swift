import CoreGraphics
import CoreML
import Foundation
import StableDiffusion
import os

private let logger = Logger(
  subsystem: "com.hongbomiao.mobile-ios",
  category: "StableDiffusionService"
)

func generateStableDiffusionImage(
  promptText: String,
  negativePromptText: String,
  seedValue: UInt32,
  stepCount: Int,
  guidanceScaleValue: Float,
  shouldDisableSafetyCheck: Bool,
  shouldReduceMemoryFootprint: Bool
) throws -> CGImage {
  // Try to find resources by looking for a known model file (TextEncoder.mlmodelc)
  // and deriving the parent directory from it
  logger.info("Looking for StableDiffusion resources")
  logger.info("Bundle resource URL: \(Bundle.main.resourceURL?.path ?? "nil")")

  guard
    let textEncoderUrl = Bundle.main.url(
      forResource: "TextEncoder",
      withExtension: "mlmodelc"
    )
  else {
    logger.error("StableDiffusion TextEncoder.mlmodelc not found in bundle")
    throw StableDiffusionGenerationError.resourceDirectoryMissing(
      directoryName: AppConfig.stableDiffusionResourceDirectoryName
    )
  }
  let resourceDirectoryUrl = textEncoderUrl.deletingLastPathComponent()
  logger.info("Found TextEncoder, using parent directory: \(resourceDirectoryUrl.path)")

  let modelConfiguration = MLModelConfiguration()
  modelConfiguration.computeUnits = .cpuAndNeuralEngine

  // Check if this is an SDXL model by looking for TextEncoder2.mlmodelc
  let textEncoder2Url = resourceDirectoryUrl.appendingPathComponent("TextEncoder2.mlmodelc")
  let isXlModel = FileManager.default.fileExists(atPath: textEncoder2Url.path)
  logger.info("Is SDXL model: \(isXlModel)")

  guard isXlModel else {
    logger.error("Non-SDXL StableDiffusion model is not supported by this build")
    throw StableDiffusionGenerationError.resourceDirectoryMissing(
      directoryName: "TextEncoder2.mlmodelc"
    )
  }

  _ = shouldDisableSafetyCheck

  // Use StableDiffusionXLPipeline for SDXL models (requires iOS 17+)
  guard #available(iOS 17.0, macOS 14.0, *) else {
    throw StableDiffusionGenerationError.unsupportedOsVersion
  }

  logger.info("Getting StableDiffusionXLPipeline from manager")
  let pipeline = try StableDiffusionPipelineManager.shared.getPipeline(
    resourceDirectoryUrl: resourceDirectoryUrl,
    modelConfiguration: modelConfiguration,
    shouldReduceMemoryFootprint: shouldReduceMemoryFootprint
  )

  var configuration = StableDiffusionXLPipeline.Configuration(prompt: promptText)
  configuration.negativePrompt = negativePromptText
  configuration.seed = seedValue
  configuration.stepCount = stepCount
  configuration.guidanceScale = guidanceScaleValue
  configuration.imageCount = 1
  let targetSize = Float32(AppConfig.stableDiffusionTargetImageDimensionPixelCount)
  configuration.targetSize = targetSize
  configuration.originalSize = targetSize

  logger.info("Generating images with XL pipeline")
  let generatedImages = try pipeline.generateImages(configuration: configuration)
  guard let firstImage = generatedImages.first, let cgImage = firstImage else {
    throw StableDiffusionGenerationError.imageNotGenerated
  }

  return cgImage
}
