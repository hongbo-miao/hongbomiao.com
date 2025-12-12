import SwiftUI
import os

private let logger = Logger(
  subsystem: "com.hongbomiao.mobile-ios",
  category: "StableDiffusion"
)

extension ContentViewModel {
  func handleGenerateStableDiffusionImageButtonTapped() {
    guard !isGeneratingStableDiffusionImage else {
      return
    }

    isGeneratingStableDiffusionImage = true
    stableDiffusionImage = nil

    Task {
      do {
        logger.info("Starting Stable Diffusion image generation")
        let generatedImage = try generateStableDiffusionImage(
          promptText: AppConfig.stableDiffusionSamplePrompt,
          negativePromptText: AppConfig.stableDiffusionNegativePromptText,
          seedValue: AppConfig.stableDiffusionSeedValue,
          stepCount: AppConfig.stableDiffusionStepCount,
          guidanceScaleValue: AppConfig.stableDiffusionGuidanceScaleValue,
          shouldDisableSafetyCheck: AppConfig.stableDiffusionShouldDisableSafetyCheck,
          shouldReduceMemoryFootprint: AppConfig.stableDiffusionShouldReduceMemoryFootprint
        )
        logger.info("Stable Diffusion image generation completed")

        await MainActor.run {
          stableDiffusionImage = generatedImage
        }
      } catch {
        logger.error("Stable Diffusion image generation failed: \(error.localizedDescription)")
      }

      await MainActor.run {
        isGeneratingStableDiffusionImage = false
      }
    }
  }
}
