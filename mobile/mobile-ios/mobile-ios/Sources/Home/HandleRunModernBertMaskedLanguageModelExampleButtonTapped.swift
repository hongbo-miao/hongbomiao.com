import SwiftUI

extension ContentViewModel {
  func handleRunModernBertMaskedLanguageModelExampleButtonTapped() {
    guard !isRunningModernBertMaskedLanguageModelExample else {
      return
    }

    isRunningModernBertMaskedLanguageModelExample = true
    modernBertPredictionDescription = ""

    Task {
      do {
        let predictionGroups = try runModernBertMaskedLanguageModelExample()
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
          isRunningModernBertMaskedLanguageModelExample = false
        }
      } catch {
        await MainActor.run {
          modernBertPredictionDescription =
            "ModernBERT example failed: \(error.localizedDescription)"
          isRunningModernBertMaskedLanguageModelExample = false
        }
      }
    }
  }
}
