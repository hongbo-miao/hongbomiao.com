import SwiftUI

extension ContentViewModel {
  func handleRunModernBertMaskedLanguageModelButtonTapped() {
    guard !isRunningModernBertMaskedLanguageModel else {
      return
    }

    isRunningModernBertMaskedLanguageModel = true
    modernBertPredictionDescription = ""

    Task {
      do {
        let predictionGroups = try runModernBertMaskedLanguageModel()
        let formattedResult =
          predictionGroups
          .enumerated()
          .map { displayIndex, group -> String in
            let lines = group.predictions.enumerated().map { index, prediction in
              let probabilityPercentage = prediction.probability * 100
              return
                "    \(index + 1). \(prediction.tokenText) (token \(prediction.tokenIdentifier)) – \(String(format: "%.2f%%", probabilityPercentage))"
            }
            return [
              "Mask \(displayIndex):",
              lines.joined(separator: "\n"),
            ]
            .joined(separator: "\n")
          }
          .joined(separator: "\n\n")

        await MainActor.run {
          modernBertPredictionDescription = formattedResult
          isRunningModernBertMaskedLanguageModel = false
        }
      } catch {
        await MainActor.run {
          modernBertPredictionDescription =
            "ModernBERT masked language model failed: \(error.localizedDescription)"
          isRunningModernBertMaskedLanguageModel = false
        }
      }
    }
  }
}
