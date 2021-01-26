import CircuitBreaker from 'opossum';
import logger from '../../log/utils/logger';
import meter from './meter';

const createCircuitBreaker = <FuncReturnType>(
  asyncFunc: (...args: string[]) => Promise<FuncReturnType>
): CircuitBreaker<string[], FuncReturnType> => {
  const breaker = new CircuitBreaker<string[], FuncReturnType>(asyncFunc, {
    timeout: 3000, // If our function takes longer than 3s, trigger a failure.
    errorThresholdPercentage: 50, // When 50% of requests fail, trip the circuit.
    resetTimeout: 5000, // After 5s, try again.
  });

  const beakerCounter = meter.createCounter(`${asyncFunc.name}BeakerCounter`);
  breaker.eventNames().forEach((eventName) => {
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    breaker.on(eventName, () => {
      logger.info({ breakerName: breaker.name, eventName }, 'createCircuitBreaker');
      const labels = { breakerName: breaker.name, eventName: String(eventName) };
      beakerCounter.bind(labels).add(1);
    });
  });
  return breaker;
};

export default createCircuitBreaker;
