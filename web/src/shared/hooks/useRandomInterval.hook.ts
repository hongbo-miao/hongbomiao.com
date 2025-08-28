import React from 'react';
import random from '@/shared/utils/random';

type Callback = () => void;

const useRandomInterval = (callback: Callback, minDelay: number, maxDelay: number): Callback => {
  const timeoutId = React.useRef(0);
  const savedCallback = React.useRef(callback);

  React.useEffect(() => {
    savedCallback.current = callback;
  });
  React.useEffect(() => {
    const handleTick = (): void => {
      const nextTickAt = random(minDelay, maxDelay);
      timeoutId.current = window.setTimeout(() => {
        savedCallback.current();
        handleTick();
      }, nextTickAt);
    };
    handleTick();
    return (): void => {
      window.clearTimeout(timeoutId.current);
    };
  }, [minDelay, maxDelay]);

  return React.useCallback(() => {
    window.clearTimeout(timeoutId.current);
  }, []);
};

export default useRandomInterval;
