import PlayerPane from '@/App/components/PlayerPane';
import fetchHlsPlaylistUrl from '@/shared/hls/utils/fetchHlsPlaylistUrl';
import useBarLatencySeconds from '@/shared/video/hooks/useBarLatencySeconds';
import Hls from 'hls.js';
import { useEffect, useRef, useState } from 'react';

const RETRY_INTERVAL_MS = 2_000;

export default function HlsPlayer() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [status, setStatus] = useState('Discovering playlist');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const latencySeconds = useBarLatencySeconds(videoRef);

  useEffect(() => {
    const videoElement = videoRef.current;
    if (videoElement == null) {
      return;
    }

    let hls: Hls | undefined;
    let retryTimeoutId: ReturnType<typeof setTimeout> | undefined;
    let isCancelled = false;

    fetchHlsPlaylistUrl()
      .then((playlistUrl) => {
        if (isCancelled) {
          return;
        }

        if (!Hls.isSupported()) {
          videoElement.src = playlistUrl;
          return;
        }

        hls = new Hls();

        function loadPlaylist() {
          hls?.loadSource(playlistUrl);
        }

        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          setStatus('Playing segments from RustFS');
          setIsPlaying(true);
          void videoElement.play();
        });

        hls.on(Hls.Events.ERROR, (_event, data) => {
          if (data.fatal) {
            setStatus('Playlist not ready yet, retrying');
            setIsPlaying(false);
            retryTimeoutId = setTimeout(loadPlaylist, RETRY_INTERVAL_MS);
          }
        });

        hls.attachMedia(videoElement);
        loadPlaylist();
      })
      .catch((error: unknown) => {
        setStatus(`Failed to discover playlist: ${String(error)}`);
      });

    return () => {
      isCancelled = true;
      if (retryTimeoutId !== undefined) {
        clearTimeout(retryTimeoutId);
      }
      hls?.destroy();
    };
  }, []);

  return (
    <PlayerPane
      title="HLS"
      subtitle="Egress transcodes to H.264/AAC, writes 2 s segments to RustFS, hls.js plays them"
      status={status}
      isPlaying={isPlaying}
      latencySeconds={latencySeconds}
      isAudioEnabled={isAudioEnabled}
      // Browsers block autoplay with sound until a user gesture, so playback starts
      // muted and this toggle is what unlocks it.
      onToggleAudio={() => {
        setIsAudioEnabled((isEnabled) => !isEnabled);
      }}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted={!isAudioEnabled}
        style={{ height: '100%', width: '100%' }}
      />
    </PlayerPane>
  );
}
