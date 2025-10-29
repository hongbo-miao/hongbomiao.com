import { Loader2, Play, Square } from 'lucide-react';
import useFireAudioStream from '@/Fire/hooks/useFireAudioStream';
import { Button } from '@/components/ui/button';

function Fire() {
  const {
    state: { isConnecting, isConnected, errorMessage, chunkCount },
    connectToFireAudioStream,
    disconnectFromFireAudioStream,
  } = useFireAudioStream();

  return (
    <div className="mx-auto flex max-w-4xl flex-col p-6">
      <div>
        <h1 className="text-3xl font-semibold">Fire</h1>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <Button
          type="button"
          size="icon"
          className="cursor-pointer disabled:cursor-not-allowed"
          onClick={() => void connectToFireAudioStream()}
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
          onClick={() => void disconnectFromFireAudioStream()}
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

export default Fire;
