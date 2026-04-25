import { LiveKitRoom, useTracks } from '@livekit/components-react';
import { Track, type RemoteTrackPublication } from 'livekit-client';
import { useEffect, useState } from 'react';
import DeviceCard from '@/App/components/DeviceCard';

interface AudioRoomProps {
  onDisconnect: () => void;
}

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

function TrackGrid() {
  const tracks = useTracks([Track.Source.Microphone], { onlySubscribed: false });
  const [activeDeviceId, setActiveDeviceId] = useState<string | null>(null);

  useEffect(() => {
    if (tracks.length > 0 && activeDeviceId == null) {
      const firstDeviceId = tracks[0]!.publication.trackName?.replace(/^audio-/, '') ?? 'unknown';
      setActiveDeviceId(firstDeviceId);
    }
  }, [tracks, activeDeviceId]);

  useEffect(() => {
    for (const trackRef of tracks) {
      const deviceId = trackRef.publication.trackName?.replace(/^audio-/, '') ?? 'unknown';
      (trackRef.publication as RemoteTrackPublication).setSubscribed(deviceId === activeDeviceId);
    }
  }, [activeDeviceId, tracks]);

  if (tracks.length === 0) {
    return <p>Waiting for audio tracks...</p>;
  }

  const trackByDeviceId = new Map(
    tracks.map((trackRef) => [
      trackRef.publication.trackName?.replace(/^audio-/, '') ?? 'unknown',
      trackRef,
    ])
  );
  const uniqueTracks = [...trackByDeviceId.values()];

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
      {uniqueTracks.map((trackRef) => {
        const deviceId =
          trackRef.publication.trackName?.replace(/^audio-/, '') ?? 'unknown';
        return (
          <DeviceCard
            key={trackRef.publication.trackSid}
            deviceId={deviceId}
            trackRef={trackRef}
            isActive={activeDeviceId === deviceId}
            onSelect={() => setActiveDeviceId(deviceId)}
          />
        );
      })}
    </div>
  );
}

export default function AudioRoom({ onDisconnect }: AudioRoomProps) {
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
    return <p>Error: {connectionError}</p>;
  }

  if (config == null) {
    return <p>Connecting...</p>;
  }

  return (
    <LiveKitRoom
      token={config.token}
      serverUrl={config.livekitUrl}
      connect={true}
      audio={false}
      video={false}
      onDisconnected={onDisconnect}
    >
      <button onClick={onDisconnect} style={{ marginBottom: '16px' }}>
        Disconnect
      </button>
      <TrackGrid />
    </LiveKitRoom>
  );
}
