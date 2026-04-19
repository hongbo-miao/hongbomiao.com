import { useState } from 'react';
import AudioRoom from '@/App/components/AudioRoom';

export default function App() {
  const [isConnected, setIsConnected] = useState(false);

  return (
    <div style={{ fontFamily: 'monospace', padding: '16px' }}>
      {isConnected ? (
        <AudioRoom onDisconnect={() => setIsConnected(false)} />
      ) : (
        <button onClick={() => setIsConnected(true)}>Connect</button>
      )}
    </div>
  );
}
