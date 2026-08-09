import type { ReactNode } from 'react';

interface PlayerPaneProps {
  title: string;
  subtitle: string;
  status: string;
  isPlaying: boolean;
  latencySeconds: number | null;
  isAudioEnabled: boolean;
  onToggleAudio: () => void;
  children: ReactNode;
}

export default function PlayerPane({
  title,
  subtitle,
  status,
  isPlaying,
  latencySeconds,
  isAudioEnabled,
  onToggleAudio,
  children,
}: PlayerPaneProps) {
  return (
    <section
      style={{
        border: '1px solid #d0d0d0',
        borderRadius: '8px',
        padding: '16px',
      }}
    >
      <h2 style={{ margin: '0 0 4px' }}>{title}</h2>
      <p style={{ color: '#666', margin: '0 0 12px' }}>{subtitle}</p>

      <div
        style={{
          alignItems: 'center',
          aspectRatio: '16 / 9',
          background: '#000',
          border: '1px solid #d0d0d0',
          display: 'flex',
          justifyContent: 'center',
          overflow: 'hidden',
          width: '100%',
        }}
      >
        {children}
      </div>

      <dl
        style={{
          display: 'grid',
          gap: '4px 12px',
          gridTemplateColumns: 'auto 1fr',
          margin: '12px 0',
        }}
      >
        <dt style={{ color: '#666' }}>Status</dt>
        <dd style={{ margin: 0 }}>
          <span style={{ color: isPlaying ? '#0a7d28' : '#a06000' }}>{isPlaying ? '●' : '○'}</span>{' '}
          {status}
        </dd>
        <dt style={{ color: '#666' }}>Latency</dt>
        <dd style={{ margin: 0 }}>
          {latencySeconds === null ? '—' : `${latencySeconds.toFixed(1)} s behind live`}
        </dd>
      </dl>

      <button onClick={onToggleAudio} style={{ padding: '6px 12px' }} type="button">
        {isAudioEnabled ? 'Mute audio' : 'Enable audio'}
      </button>
    </section>
  );
}
