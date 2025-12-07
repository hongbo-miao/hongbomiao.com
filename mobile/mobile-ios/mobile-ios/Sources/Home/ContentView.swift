import SwiftUI
@preconcurrency import WhisperKit

struct ContentView: View {
  @StateObject private var contentViewModel: ContentViewModel = ContentViewModel()

  var body: some View {
    VStack(spacing: 24) {
      Button(
        action: {
          contentViewModel.handleRealTimeAudioTranscriptionButtonTapped()
        },
        label: {
          if contentViewModel.isStreamingAudioTranscription {
            Text("Stop Live Transcription")
          } else {
            Text("Start Live Transcription")
          }
        }
      )
      .buttonStyle(.borderedProminent)

      if !contentViewModel.streamingTranscribedText.isEmpty {
        ScrollView {
          Text(contentViewModel.streamingTranscribedText)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
      }

      Divider()

      Button(
        action: {
          contentViewModel.handleTranscribeAudioButtonTapped()
        },
        label: {
          if contentViewModel.isTranscribingAudio {
            ProgressView()
          } else {
            Text("Transcribe Audio")
          }
        }
      )
      .buttonStyle(.borderedProminent)

      if !contentViewModel.transcribedText.isEmpty {
        ScrollView {
          Text(contentViewModel.transcribedText)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
      }

      Divider()

      Button(
        action: {
          contentViewModel.handleGenerateJokeButtonTapped()
        },
        label: {
          if contentViewModel.isGeneratingJoke {
            ProgressView()
          } else {
            Text("Generate Joke")
          }
        }
      )
      .buttonStyle(.borderedProminent)

      if !contentViewModel.jokeText.isEmpty {
        ScrollView {
          Text(contentViewModel.jokeText)
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
