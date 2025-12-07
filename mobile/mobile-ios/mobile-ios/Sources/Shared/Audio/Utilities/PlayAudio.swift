import AVFoundation
import Foundation

@MainActor
private var audioPlayer: AVAudioPlayer?

@MainActor
func playAudio(audioResourceName: String, audioResourceExtension: String) throws {
  guard
    let audioUrl = Bundle.main.url(
      forResource: audioResourceName,
      withExtension: audioResourceExtension
    )
  else {
    throw AudioPlaybackError.audioFileNotFound(
      name: audioResourceName,
      extension: audioResourceExtension
    )
  }

  do {
    try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
    try AVAudioSession.sharedInstance().setActive(true)
    audioPlayer = try AVAudioPlayer(contentsOf: audioUrl)
    audioPlayer?.play()
  } catch {
    throw AudioPlaybackError.playbackFailed(error)
  }
}
