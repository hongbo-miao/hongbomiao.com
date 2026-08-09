import PlayerPane from '@/App/components/PlayerPane';
import useBarLatencySeconds from '@/shared/video/hooks/useBarLatencySeconds';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useRoomContext,
  useTracks,
} from '@livekit/components-react';
import { Track } from 'livekit-client';
import { useEffect, useRef, useState } from 'react';

interface ConnectionConfig {
  token: string;
  livekitUrl: string;
}

async function fetchConnectionConfig(): Promise<ConnectionConfig> {
  const identity = `viewer-${Date.now()}`;
  const response = await fetch(`/token?identity=${identity}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch token: ${response.statusText}`);
  }
  const data = (await response.json()) as { token: string; livekit_url: string };
  return { token: data.token, livekitUrl: data.livekit_url };
}

function SubscribedPane() {
  const room = useRoomContext();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const latencySeconds = useBarLatencySeconds(videoRef);
  const videoTrack = useTracks([Track.Source.Camera])[0]?.publication.videoTrack ?? null;

  useEffect(() => {
    const videoElement = videoRef.current;
    if (videoTrack == null || videoElement == null) {
      return;
    }
    videoTrack.attach(videoElement);
    return () => {
      videoTrack.detach(videoElement);
    };
  }, [videoTrack]);

  useEffect(() => {
    for (const participant of room.remoteParticipants.values()) {
      participant.setVolume(isAudioEnabled ? 1 : 0);
    }
  }, [isAudioEnabled, room]);

  return (
    <PlayerPane
      title="WebRTC"
      subtitle="Subscribed straight from the LiveKit SFU, no transcode, no segments"
      status={videoTrack == null ? 'Connected, waiting for the video track' : 'Subscribed to track'}
      isPlaying={videoTrack != null}
      latencySeconds={latencySeconds}
      isAudioEnabled={isAudioEnabled}
      // The SFU audio also needs a user gesture before the browser will play it, so the
      // same toggle drives Room.startAudio and the per-participant volume.
      onToggleAudio={() => {
        const willEnable = !isAudioEnabled;
        setIsAudioEnabled(willEnable);
        if (willEnable) {
          void room.startAudio();
        }
      }}
    >
      <video ref={videoRef} autoPlay playsInline muted style={{ height: '100%', width: '100%' }} />
      <RoomAudioRenderer muted={!isAudioEnabled} />
    </PlayerPane>
  );
}

export default function WebrtcPlayer() {
  const [config, setConfig] = useState<ConnectionConfig | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  useEffect(() => {
    fetchConnectionConfig()
      .then(setConfig)
      .catch((error: unknown) => {
        setConnectionError(String(error));
      });
  }, []);

  if (connectionError !== null) {
    return (
      <PlayerPane
        title="WebRTC"
        subtitle="Subscribed straight from the LiveKit SFU, no transcode, no segments"
        status={connectionError}
        isPlaying={false}
        latencySeconds={null}
        isAudioEnabled={false}
        onToggleAudio={() => {}}
      >
        <span style={{ color: '#888' }}>No signal</span>
      </PlayerPane>
    );
  }

  if (config == null) {
    return (
      <PlayerPane
        title="WebRTC"
        subtitle="Subscribed straight from the LiveKit SFU, no transcode, no segments"
        status="Fetching a viewer token"
        isPlaying={false}
        latencySeconds={null}
        isAudioEnabled={false}
        onToggleAudio={() => {}}
      >
        <span style={{ color: '#888' }}>No signal</span>
      </PlayerPane>
    );
  }

  return (
    <LiveKitRoom
      token={config.token}
      serverUrl={config.livekitUrl}
      connect={true}
      audio={false}
      video={false}
    >
      <SubscribedPane />
    </LiveKitRoom>
  );
}
