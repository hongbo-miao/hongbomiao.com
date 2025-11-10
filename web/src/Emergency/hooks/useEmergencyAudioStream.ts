import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export type EmergencyAudioStreamState = {
  isConnecting: boolean;
  isConnected: boolean;
  errorMessage: string | null;
  chunkCount: number;
};

type EmergencyAudioStreamControls = {
  state: EmergencyAudioStreamState;
  connectToEmergencyAudioStream: () => Promise<void>;
  disconnectFromEmergencyAudioStream: () => Promise<void>;
};

const pcmSampleRateHz = 16000;
const pcmChannelCount = 1;
const initialBufferDelaySeconds = 0.5;
const maxBufferAheadSeconds = 2.0;

function useEmergencyAudioStream(): EmergencyAudioStreamControls {
  const [state, setState] = useState<EmergencyAudioStreamState>({
    isConnecting: false,
    isConnected: false,
    errorMessage: null,
    chunkCount: 0,
  });
  const transportRef = useRef<WebTransport | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlaybackTimeRef = useRef<number>(0);

  useEffect(() => {
    return () => {
      void (async () => {
        if (transportRef.current) {
          try {
            await transportRef.current.close();
          } catch {
            // intentionally ignore close errors during cleanup
          }
        }

        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          await audioContextRef.current.close();
        }

        transportRef.current = null;
        audioContextRef.current = null;
        nextPlaybackTimeRef.current = 0;
      })();
    };
  }, []);

  const ensureAudioContext = useCallback(async (): Promise<AudioContext> => {
    let context = audioContextRef.current;
    if (context == null || context.state === 'closed') {
      context = new AudioContext({ sampleRate: pcmSampleRateHz });
      audioContextRef.current = context;
      nextPlaybackTimeRef.current = context.currentTime;
    }

    if (context.state === 'suspended') {
      await context.resume();
    }

    return context;
  }, []);

  const scheduleChunk = useCallback(
    async (chunk: Uint8Array) => {
      const context = await ensureAudioContext();
      const int16 = new Int16Array(chunk.buffer, chunk.byteOffset, chunk.byteLength / 2);
      const float32 = new Float32Array(int16.length);

      for (let index = 0; index < int16.length; index += 1) {
        const sample = int16[index] / 32768;
        float32[index] = Math.max(-1, Math.min(1, sample));
      }

      const buffer = context.createBuffer(pcmChannelCount, float32.length, pcmSampleRateHz);
      buffer.copyToChannel(float32, 0);

      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);

      if (nextPlaybackTimeRef.current === 0 || nextPlaybackTimeRef.current < context.currentTime) {
        nextPlaybackTimeRef.current = context.currentTime + initialBufferDelaySeconds;
      }

      const startTime = nextPlaybackTimeRef.current;

      if (startTime - context.currentTime > maxBufferAheadSeconds) {
        console.warn(`Dropping chunk: buffer ahead by ${(startTime - context.currentTime).toFixed(2)} seconds`);
        return;
      }

      source.start(startTime);
      nextPlaybackTimeRef.current = startTime + buffer.duration;

      source.onended = () => {
        if (context.currentTime > nextPlaybackTimeRef.current + 0.5) {
          nextPlaybackTimeRef.current = context.currentTime + initialBufferDelaySeconds;
        }
      };
    },
    [ensureAudioContext],
  );

  const connectToEmergencyAudioStream = useCallback(async () => {
    if (!('WebTransport' in window)) {
      setState((previousState) => ({
        ...previousState,
        errorMessage: 'WebTransport is not supported in this browser.',
      }));
      return;
    }

    setState({ isConnecting: true, isConnected: false, errorMessage: null, chunkCount: 0 });

    const currentOrigin = window.location.origin;
    const defaultSecureOrigin = currentOrigin.replace('http://', 'https://');
    const defaultWebTransportOrigin = defaultSecureOrigin.replace(/:\d+$/, ':36148');
    const serverUrl = `${defaultWebTransportOrigin}/emergency-audio-stream/lincoln.fire`;

    try {
      const webTransport = new WebTransport(serverUrl);
      transportRef.current = webTransport;

      await webTransport.ready;

      setState((previousState) => ({
        ...previousState,
        isConnecting: false,
        isConnected: true,
      }));

      const reader = webTransport.incomingUnidirectionalStreams.getReader();

      while (true) {
        const { value: stream, done } = await reader.read();
        if (done || !stream) {
          break;
        }

        const streamReader = stream.getReader();
        let frameBuffer = new Uint8Array();

        while (true) {
          const { value: chunk, done: isStreamDone } = await streamReader.read();
          if (isStreamDone || !chunk) {
            break;
          }

          const newData = new Uint8Array(frameBuffer.length + chunk.length);
          newData.set(frameBuffer);
          newData.set(chunk, frameBuffer.length);
          frameBuffer = newData;

          while (frameBuffer.length >= 4) {
            const frameLength = new DataView(frameBuffer.buffer, frameBuffer.byteOffset, 4).getUint32(0, true);

            if (frameBuffer.length >= 4 + frameLength) {
              const pcmChunk = frameBuffer.slice(4, 4 + frameLength);
              frameBuffer = frameBuffer.slice(4 + frameLength);

              await scheduleChunk(pcmChunk);
              setState((previousState) => ({
                ...previousState,
                chunkCount: previousState.chunkCount + 1,
              }));
            } else {
              break;
            }
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to connect to Lincoln Emergency audio stream.';
      setState({ isConnecting: false, isConnected: false, errorMessage: message, chunkCount: 0 });
      transportRef.current = null;
    }
  }, [scheduleChunk]);

  const disconnectFromEmergencyAudioStream = useCallback(async () => {
    setState((previousState) => ({
      ...previousState,
      isConnecting: false,
      isConnected: false,
    }));

    if (transportRef.current) {
      try {
        await transportRef.current.close();
      } catch {
        // intentionally ignore close errors during disconnect
      }
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      await audioContextRef.current.close();
    }

    transportRef.current = null;
    audioContextRef.current = null;
    nextPlaybackTimeRef.current = 0;

    setState((previousState) => ({
      ...previousState,
      chunkCount: 0,
    }));
  }, []);

  return useMemo(
    () => ({
      state,
      connectToEmergencyAudioStream,
      disconnectFromEmergencyAudioStream,
    }),
    [connectToEmergencyAudioStream, disconnectFromEmergencyAudioStream, state],
  );
}

export default useEmergencyAudioStream;
