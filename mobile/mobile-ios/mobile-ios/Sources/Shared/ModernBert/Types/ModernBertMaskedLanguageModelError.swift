import Foundation

enum ModernBertMaskedLanguageModelError: LocalizedError {
  case modelNotFound
  case tokenizerDirectoryMissing
  case tokenizerFilesMissing
  case maskTokenIdentifierMissing
  case maskTokenNotPresent
  case logitsUnavailable

  var errorDescription: String? {
    switch self {
    case .modelNotFound:
      return "ModernBERT model package is missing from the bundle."
    case .tokenizerDirectoryMissing:
      return "ModernBERT tokenizer directory is missing from the bundle."
    case .tokenizerFilesMissing:
      return "ModernBERT tokenizer JSON files are missing from the bundle."
    case .maskTokenIdentifierMissing:
      return "ModernBERT tokenizer does not define a mask token identifier."
    case .maskTokenNotPresent:
      return "ModernBERT sample sentence does not contain a mask token."
    case .logitsUnavailable:
      return "ModernBERT prediction logits are unavailable."
    }
  }
}
