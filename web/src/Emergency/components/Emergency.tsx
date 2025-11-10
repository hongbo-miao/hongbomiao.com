import { Loader2, Play, Square } from 'lucide-react';
import useEmergencyAudioStream from '@/Emergency/hooks/useEmergencyAudioStream';
import { Button } from '@/components/ui/button';

function Emergency() {
  const {
    state: { isConnecting, isConnected, errorMessage, chunkCount },
    connectToEmergencyAudioStream,
    disconnectFromEmergencyAudioStream,
  } = useEmergencyAudioStream();

  return (
    <div className="mx-auto flex max-w-4xl flex-col p-6">
      <div>
        <h1 className="text-3xl font-semibold">Emergency</h1>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <Button
          type="button"
          size="icon"
          className="cursor-pointer disabled:cursor-not-allowed"
          onClick={() => void connectToEmergencyAudioStream()}
          disabled={isConnecting || isConnected}
        >
          {isConnecting ? (
            <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
          ) : (
            <Play aria-hidden="true" className="h-4 w-4" />
          )}
        </Button>
        <Button
          type="button"
          size="icon"
          variant="outline"
          className="cursor-pointer disabled:cursor-not-allowed"
          onClick={() => void disconnectFromEmergencyAudioStream()}
          disabled={!isConnected}
        >
          <Square aria-hidden="true" className="h-4 w-4" />
        </Button>
        <span className="text-sm text-muted-foreground">Chunks received: {chunkCount}</span>
      </div>

      {errorMessage ? <div className="text-sm text-destructive">{errorMessage}</div> : null}
    </div>
  );
}

export default Emergency;
