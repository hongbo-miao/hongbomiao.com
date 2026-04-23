import { useDataChannel, useRoomContext } from '@livekit/components-react';
import type { TrackReferenceOrPlaceholder } from '@livekit/components-react';
import type { RemoteAudioTrack } from 'livekit-client';
import { useEffect, useRef, useState } from 'react';

interface DeviceCardProps {
  deviceId: string;
  trackRef: TrackReferenceOrPlaceholder;
  isActive: boolean;
  onSelect: () => void;
}

const CANVAS_WIDTH = 240;
const CANVAS_HEIGHT = 60;
const FFT_SIZE = 2048;

let sharedAudioContext: AudioContext | null = null;

function getAudioContext(): AudioContext {
  if (!sharedAudioContext || sharedAudioContext.state === 'closed') {
    sharedAudioContext = new AudioContext();
  }
  return sharedAudioContext;
}

export default function DeviceCard({ deviceId, trackRef, isActive, onSelect }: DeviceCardProps) {
  const room = useRoomContext();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const minRawLatencyRef = useRef<number>(Infinity);
  const [transcriptLines, setTranscriptLines] = useState<string[]>([]);
  const [interimText, setInterimText] = useState<string | null>(null);

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
      setLatencyMs((prev) => Math.round(prev === null ? sample : prev * 0.95 + sample * 0.05));
    }
  });

  useEffect(() => {
    if (!isActive) return;
    const track = trackRef.publication?.track as RemoteAudioTrack | undefined;
    if (!track || !canvasRef.current) return;

    // Minimize WebRTC jitter buffer
    if (track.mediaStreamTrack) {
      try {
        const engine = (room as any).engine;
        const peerConnection: RTCPeerConnection | undefined = engine?.pcManager?.publisher?.pc;
        if (peerConnection) {
          const receiver = peerConnection.getReceivers().find((r) => r.track.id === track.mediaStreamTrack.id);
          if (receiver) {
            const r = receiver as any;
            // jitterBufferTarget is the current standard; playoutDelayHint is the older alias
            if ('jitterBufferTarget' in r) {
              r.jitterBufferTarget = 0;
            } else if ('playoutDelayHint' in r) {
              r.playoutDelayHint = 0;
            }
          }
        }
      } catch (error) {
        console.warn('[playoutDelayHint] failed:', error);
      }
    }

    const audioContext = getAudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = FFT_SIZE;

    track.setAudioContext(audioContext);
    track.setWebAudioPlugins([analyser]);

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
      cancelAnimationFrame(animationId);
      audioContext.removeEventListener('statechange', handleStateChange);
      track.detach(audioElement);
      track.setWebAudioPlugins([]);
    };
  }, [trackRef.publication?.track, isActive, room]);

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
      <div style={{ marginBottom: '4px', fontSize: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ color: isActive ? '#00aa00' : '#aaa' }}>
          {isActive ? '▶ ' : ''}{deviceId}
        </span>
        {isActive && <span style={{ color: '#888' }}>{latencyMs !== null ? `${latencyMs}ms` : '—'}</span>}
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
