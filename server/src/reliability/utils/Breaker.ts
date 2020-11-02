import DataLoader from 'dataloader';
import CircuitBreaker from 'opossum';
import meter from '../../log/utils/meter';

class Breaker<FuncReturnType> {
  private readonly name: string;

  private readonly breaker: CircuitBreaker<string[], FuncReturnType>;

  constructor(name: string, asyncFunc: (...args: string[]) => Promise<FuncReturnType>) {
    this.name = name;
    this.breaker = new CircuitBreaker<string[], FuncReturnType>(asyncFunc, {
      timeout: 3000, // If our function takes longer than 3s, trigger a failure.
      errorThresholdPercentage: 50, // When 50% of requests fail, trip the circuit.
      resetTimeout: 5000, // After 5s, try again.
    });
    this.initCounter();
  }

  private initCounter = (): void => {
    const beakerCounter = meter.createCounter(`${this.name}BeakerCounter`);
    this.breaker.eventNames().forEach((eventName) => {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      this.breaker.on(eventName, () => {
        const labels = { breakerName: this.breaker.name, eventName: String(eventName) };
        beakerCounter.bind(labels).add(1);
      });
    });
  };

  public fire = async (id: string, dataLoader: DataLoader<string, FuncReturnType>): Promise<FuncReturnType> => {
    return this.breaker
      .fire(id)
      .then((res) => res)
      .catch((err) => {
        dataLoader.clear(id);
        return err;
      });
  };
}

export default Breaker;
