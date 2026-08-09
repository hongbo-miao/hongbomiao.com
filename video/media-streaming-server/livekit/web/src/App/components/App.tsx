import HlsPlayer from '@/App/components/HlsPlayer';
import WebrtcPlayer from '@/App/components/WebrtcPlayer';

export default function App() {
  return (
    <div style={{ fontFamily: 'monospace', margin: '0 auto', maxWidth: '1600px', padding: '24px' }}>
      <h1 style={{ marginBottom: '4px' }}>LiveKit: WebRTC vs HLS</h1>
      <p style={{ color: '#666', marginTop: 0 }}>
        One publisher sends a single audio track (a 440&nbsp;Hz tone that beeps every second) and a
        single video track (a bar that steps once per second). Both panes below carry that same pair
        of tracks, delivered two different ways.
      </p>
      <p style={{ color: '#666', marginTop: 0 }}>
        The bar position encodes the wall-clock second the frame was drawn, so each pane decodes it
        and reports how far behind live it is. Expect up to ~300&nbsp;ms for WebRTC and ~5-30&nbsp;s
        for HLS.
      </p>
      <div
        style={{
          display: 'grid',
          gap: '24px',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          marginTop: '24px',
        }}
      >
        <WebrtcPlayer />
        <HlsPlayer />
      </div>
    </div>
  );
}
