import SwiftUI
@preconcurrency import WhisperKit

struct ContentView: View {
  @State private var isTranscribingAudio: Bool = false
  @State private var transcriptionStatusMessage: String = ""
  @State private var transcribedText: String = ""

  private let audioResourceName: String = "audio"
  private let audioResourceExtension: String = "wav"

  var body: some View {
    VStack(spacing: 24) {
      Button(
        action: {
          guard !isTranscribingAudio else {
            return
          }

          isTranscribingAudio = true
          transcriptionStatusMessage = ""
          transcribedText = ""

          Task {
            let transcriptionOutcome = await transcribeAudio(
              audioResourceName: audioResourceName,
              audioResourceExtension: audioResourceExtension
            )

            await MainActor.run {
              transcribedText = transcriptionOutcome.transcribedText ?? ""
              transcriptionStatusMessage = transcriptionOutcome.statusMessage
              isTranscribingAudio = false
            }
          }
        },
        label: {
          if isTranscribingAudio {
            ProgressView()
          } else {
            Text("Transcribe Audio")
          }
        }
      )
      .buttonStyle(.borderedProminent)

      if !transcriptionStatusMessage.isEmpty {
        Text(transcriptionStatusMessage)
          .multilineTextAlignment(.center)
      }

      if !transcribedText.isEmpty {
        ScrollView {
          Text(transcribedText)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
      }
    }
    .padding()
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
