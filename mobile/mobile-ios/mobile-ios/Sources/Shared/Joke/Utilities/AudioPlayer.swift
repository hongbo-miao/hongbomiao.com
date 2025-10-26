import AVFoundation
import Foundation

final class AudioPlayer: NSObject, @unchecked Sendable {
  private let queue = DispatchQueue(label: "com.hongbomiao.AudioPlayer")
  private var audioEngine: AVAudioEngine?
  private var playerNode: AVAudioPlayerNode?
  private var continuation: CheckedContinuation<Void, Never>?

  func playAudio(samples: [Float], sampleRate: Double) async {
    await withCheckedContinuation { continuation in
      queue.async { [weak self] in
        guard let self else {
          continuation.resume()
          return
        }

        self.continuation = continuation

        let frameCount = AVAudioFrameCount(samples.count)

        // Create audio format: mono, 32-bit float PCM
        guard
          let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
          ),
          let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
          ),
          let channelData = buffer.floatChannelData?[0]
        else {
          continuation.resume()
          return
        }

        // Set buffer length and copy samples
        buffer.frameLength = frameCount

        _ = UnsafeMutableBufferPointer(start: channelData, count: Int(frameCount)).initialize(
          from: samples)

        // Setup audio engine
        let audioEngine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()

        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)

        self.audioEngine = audioEngine
        self.playerNode = playerNode

        do {
          try audioEngine.start()

          // Schedule buffer and set completion handler
          playerNode.scheduleBuffer(buffer) { [weak self] in
            guard let self else { return }
            self.queue.async {
              self.cleanup()
            }
          }

          playerNode.play()
        } catch {
          self.cleanup()
        }
      }
    }
  }

  private func cleanup() {
    dispatchPrecondition(condition: .onQueue(queue))

    audioEngine?.stop()
    audioEngine = nil
    playerNode = nil
    continuation?.resume()
    continuation = nil
  }
}
