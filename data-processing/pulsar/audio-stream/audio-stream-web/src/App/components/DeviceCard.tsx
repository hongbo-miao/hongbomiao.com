import { useDataChannel, useRoomContext } from '@livekit/components-react';
import type { TrackReferenceOrPlaceholder } from '@livekit/components-react';
import type { RemoteAudioTrack } from 'livekit-client';
import { useEffect, useRef, useState } from 'react';
import { createDenoiseNode } from '@/App/utils/createDenoiseNode';
import { getRnnoiseInstance, loadRnnoise } from '@/App/utils/loadRnnoise';

interface DeviceCardProps {
  deviceId: string;
  trackRef: TrackReferenceOrPlaceholder;
  isActive: boolean;
  onSelect: () => void;
}

const CANVAS_WIDTH = 360;
const CANVAS_HEIGHT = 60;
const FAST_FOURIER_TRANSFORM_SIZE = 2048;
const SOUND_PROPAGATION_MS = Math.round(1000 / 343);

let sharedAudioContext: AudioContext | null = null;

function getAudioContext(): AudioContext {
  if (!sharedAudioContext || sharedAudioContext.state === 'closed') {
    sharedAudioContext = new AudioContext({ sampleRate: 48000 });
  }
  return sharedAudioContext;
}

export default function DeviceCard({ deviceId, trackRef, isActive, onSelect }: DeviceCardProps) {
  const room = useRoomContext();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const minRawLatencyRef = useRef<number>(Infinity);
  const [jitterBufferMs, setJitterBufferMs] = useState<number | null>(null);
  const [audioOutputLatencyMs, setAudioOutputLatencyMs] = useState<number | null>(null);
  const [transcriptLines, setTranscriptLines] = useState<string[]>([]);
  const [interimText, setInterimText] = useState<string | null>(null);
  const [isNoiseCancellationEnabled, setIsNoiseCancellationEnabled] = useState(false);
  const [isRnnoiseReady, setIsRnnoiseReady] = useState(() => getRnnoiseInstance() !== null);

  useDataChannel('transcript', (msg) => {
    const data = JSON.parse(new TextDecoder().decode(msg.payload)) as {
      device_id: string;
      text: string;
      is_final?: boolean;
    };
    if (data.device_id === deviceId) {
      if (data.is_final === false) {
        setInterimText(data.text);
      } else {
        setInterimText(null);
        setTranscriptLines((prev) => [...prev.slice(-49), data.text]);
      }
    }
  });

  useDataChannel('audio-seq', (msg) => {
    const text = new TextDecoder().decode(msg.payload);
    const data = JSON.parse(text) as { device_id: string; timestamp_ns: number };
    if (data.device_id === deviceId) {
      const raw = Date.now() - data.timestamp_ns / 1_000_000;
      if (raw < minRawLatencyRef.current) minRawLatencyRef.current = raw;
      const sample = raw - minRawLatencyRef.current;
      setLatencyMs((prev) => Math.round(prev == null ? sample : prev * 0.95 + sample * 0.05));
    }
  });

  useEffect(() => {
    if (!isActive) return;
    loadRnnoise()
      .then(() => setIsRnnoiseReady(true))
      .catch(() => undefined);
  }, [isActive]);

  useEffect(() => {
    if (!isActive) return;
    const track = trackRef.publication?.track as RemoteAudioTrack | undefined;
    if (!track || !canvasRef.current) return;

    // Minimize WebRTC jitter buffer
    let statsIntervalId: ReturnType<typeof setInterval> | null = null;
    if (track.mediaStreamTrack) {
      try {
        const engine = (room as unknown as { engine: { pcManager?: { publisher?: { pc: RTCPeerConnection } } } }).engine;
        const peerConnection: RTCPeerConnection | undefined = engine?.pcManager?.publisher?.pc;
        if (peerConnection) {
          const receiver = peerConnection.getReceivers().find((r) => r.track.id === track.mediaStreamTrack.id);
          if (receiver) {
            const r = receiver as unknown as Record<string, number>;
            // jitterBufferTarget is the current standard; playoutDelayHint is the older alias
            if ('jitterBufferTarget' in r) {
              r['jitterBufferTarget'] = 0;
            } else if ('playoutDelayHint' in r) {
              r['playoutDelayHint'] = 0;
            }

            let prevJitterBufferDelay = 0;
            let prevJitterBufferEmittedCount = 0;
            statsIntervalId = setInterval(() => {
              setAudioOutputLatencyMs(Math.round((audioContext.outputLatency || audioContext.baseLatency) * 1000));
              receiver.getStats().then((statsReport) => {
                statsReport.forEach((report) => {
                  if (report.type !== 'inbound-rtp') return;
                  const inbound = report as RTCInboundRtpStreamStats & { jitterBufferDelay?: number; jitterBufferEmittedCount?: number };
                  const currentDelay = inbound.jitterBufferDelay ?? 0;
                  const currentEmittedCount = inbound.jitterBufferEmittedCount ?? 0;
                  const deltaDelay = currentDelay - prevJitterBufferDelay;
                  const deltaEmittedCount = currentEmittedCount - prevJitterBufferEmittedCount;
                  prevJitterBufferDelay = currentDelay;
                  prevJitterBufferEmittedCount = currentEmittedCount;
                  if (deltaEmittedCount > 0) {
                    setJitterBufferMs(Math.round((deltaDelay / deltaEmittedCount) * 1000));
                  }
                });
              }).catch(() => undefined);
            }, 1000);
          }
        }
      } catch (error) {
        console.warn('[playoutDelayHint] failed:', error);
      }
    }

    const audioContext = getAudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = FAST_FOURIER_TRANSFORM_SIZE;

    track.setAudioContext(audioContext);

    const rnnoise = getRnnoiseInstance();
    let denoiseNode: ScriptProcessorNode | null = null;
    if (isNoiseCancellationEnabled && rnnoise) {
      denoiseNode = createDenoiseNode(audioContext, rnnoise);
      track.setWebAudioPlugins([denoiseNode, analyser]);
    } else {
      track.setWebAudioPlugins([analyser]);
    }

    const audioElement = document.createElement('audio');
    track.attach(audioElement);

    const buffer = new Uint8Array(analyser.fftSize);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    let animationId = 0;

    function drawWaveform() {
      animationId = requestAnimationFrame(drawWaveform);
      analyser.getByteTimeDomainData(buffer);

      ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      ctx.strokeStyle = '#00aa00';
      ctx.lineWidth = 1;
      ctx.beginPath();
      const sliceWidth = CANVAS_WIDTH / buffer.length;
      let x = 0;
      for (let i = 0; i < buffer.length; i++) {
        const y = ((buffer[i]! / 128.0) - 1.0) * (CANVAS_HEIGHT / 2) + CANVAS_HEIGHT / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.stroke();
    }

    function handleStateChange() {
      if (audioContext.state === 'suspended') {
        audioContext.resume().catch(() => undefined);
      }
    }
    audioContext.addEventListener('statechange', handleStateChange);
    audioContext.resume().then(() => drawWaveform()).catch(() => undefined);

    return () => {
      if (statsIntervalId !== null) clearInterval(statsIntervalId);
      cancelAnimationFrame(animationId);
      audioContext.removeEventListener('statechange', handleStateChange);
      track.detach(audioElement);
      if (denoiseNode) {
        (denoiseNode as unknown as { _destroyDenoiseState: () => void })._destroyDenoiseState();
      }
      track.setWebAudioPlugins([]);
    };
  }, [trackRef.publication?.track, isActive, room, isNoiseCancellationEnabled, isRnnoiseReady]);

  const cardStyle: React.CSSProperties = {
    border: isActive ? '2px solid #00aa00' : '1px solid #555',
    padding: '8px',
    width: `${CANVAS_WIDTH}px`,
    cursor: isActive ? 'default' : 'pointer',
    background: isActive ? '#0a1a0a' : '#111',
    borderRadius: '4px',
  };

  return (
    <div style={cardStyle} onClick={isActive ? undefined : onSelect}>
      <div style={{ marginBottom: '4px', fontSize: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'stretch' }}>
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <span style={{ color: isActive ? '#00aa00' : '#aaa' }}>
            {deviceId}
          </span>
          {isActive && (
            <button
              onClick={() => setIsNoiseCancellationEnabled((prev) => !prev)}
              disabled={!isRnnoiseReady}
              style={{
                fontSize: '11px',
                padding: '2px 8px',
                background: isNoiseCancellationEnabled ? '#004400' : '#222',
                color: isNoiseCancellationEnabled ? '#00aa00' : '#aaa',
                border: `1px solid ${isNoiseCancellationEnabled ? '#00aa00' : '#555'}`,
                borderRadius: '3px',
                cursor: isRnnoiseReady ? 'pointer' : 'wait',
                alignSelf: 'flex-start',
              }}
            >
              {isRnnoiseReady
                ? isNoiseCancellationEnabled
                  ? 'Noise cancellation: on'
                  : 'Noise cancellation: off'
                : 'Noise cancellation: loading…'}
            </button>
          )}
        </div>
        {isActive && (
          <span style={{ color: '#888', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px', fontSize: '10px' }}>
            <span title="data channel transit time (floor + variance)">{latencyMs !== null ? `Network transit: ${Math.max(0, Math.round(minRawLatencyRef.current)) + latencyMs}ms` : 'Network transit: —'}</span>
            <span title="WebRTC receiver jitter buffer delay">{jitterBufferMs !== null ? `WebRTC jitter buffer: ${jitterBufferMs}ms` : 'WebRTC jitter buffer: —'}</span>
            <span title="Web Audio hardware output latency (AudioContext.outputLatency)">{audioOutputLatencyMs !== null ? `Speaker output latency: ${audioOutputLatencyMs}ms` : 'Speaker output latency: —'}</span>
            <span title="sound travel time from speaker to ear at 1 meter (343 m/s)">{`Sound propagation (1m): ${SOUND_PROPAGATION_MS}ms`}</span>
            {latencyMs !== null && jitterBufferMs !== null && audioOutputLatencyMs !== null && minRawLatencyRef.current !== Infinity && (
              <span title="total estimated latency from publish to ear" style={{ color: '#aaa', borderTop: '1px solid #444', paddingTop: '2px', marginTop: '2px' }}>
                Total: {Math.max(0, Math.round(minRawLatencyRef.current)) + latencyMs + jitterBufferMs + audioOutputLatencyMs + SOUND_PROPAGATION_MS}ms
              </span>
            )}
          </span>
        )}
      </div>
      {isActive && (
        <>
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            style={{ background: '#000', display: 'block' }}
          />
          <div style={{ maxHeight: '1000px', overflowY: 'auto', fontSize: '11px', color: '#aaa', marginTop: '4px' }}>
            {transcriptLines.map((line, index) => (
              <div key={index} style={{ marginBottom: '8px' }}>{line}</div>
            ))}
            {interimText && (
              <div style={{ color: '#555', fontStyle: 'italic', marginBottom: '8px' }}>{interimText}</div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
