import decodeBarEpochSecond from '@/shared/video/utils/decodeBarEpochSecond';
import { type RefObject, useEffect, useRef, useState } from 'react';

const SAMPLE_INTERVAL_MS = 100;

export default function useBarLatencySeconds(
  videoRef: RefObject<HTMLVideoElement | null>,
): number | null {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [latencySeconds, setLatencySeconds] = useState<number | null>(null);

  useEffect(() => {
    canvasRef.current ??= document.createElement('canvas');

    const intervalId = setInterval(() => {
      const videoElement = videoRef.current;
      const canvasElement = canvasRef.current;
      if (videoElement == null || canvasElement == null) {
        return;
      }

      const remoteEpochSecond = decodeBarEpochSecond(videoElement, canvasElement);
      if (remoteEpochSecond == null) {
        setLatencySeconds(null);
        return;
      }

      // The bar only resolves whole seconds, but the local clock advances
      // continuously, so comparing against it recovers sub-second precision: latency
      // grows smoothly until the next bar step is decoded, then drops back down.
      const latency = Date.now() / 1_000 - remoteEpochSecond;
      setLatencySeconds(Math.round(latency * 10) / 10);
    }, SAMPLE_INTERVAL_MS);

    return () => {
      clearInterval(intervalId);
    };
  }, [videoRef]);

  return latencySeconds;
}
