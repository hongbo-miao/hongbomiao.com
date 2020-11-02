import axios from 'axios';
import CircuitBreaker from 'opossum';
import config from '../../../config';
// eslint-disable-next-line import/no-cycle
import planetDataLoader from '../../../graphQL/dataLoaders/planet.dataLoader';
import GraphQLPlanet from '../../../graphQL/types/GraphQLPlanet.type';
import meter from '../../../log/utils/meter';
import formatPlanet from './formatPlanet';

const fetchPlanetByID = async (id: string): Promise<GraphQLPlanet | null> => {
  const { data: swapiPlanet } = await axios.get(`${config.swapiURL}/api/planets/${id}/`);
  return formatPlanet(id, swapiPlanet);
};

const breaker = new CircuitBreaker(fetchPlanetByID, {
  timeout: 3000, // If our function takes longer than 3s, trigger a failure.
  errorThresholdPercentage: 50, // When 50% of requests fail, trip the circuit.
  resetTimeout: 5000, // After 5s, try again.
});

const planetBeakerCounter = meter.createCounter('planetBeakerCounter');

breaker.eventNames().forEach((eventName) => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  breaker.on(eventName, () => {
    const labels = { breakerName: breaker.name, eventName: String(eventName) };
    planetBeakerCounter.bind(labels).add(1);
  });
});

const fetchPlanetByIDWithBreaker = async (id: string): Promise<GraphQLPlanet | null> => {
  return breaker
    .fire(id)
    .then((res) => res)
    .catch((err) => {
      planetDataLoader.clear(id);
      return err;
    });
};

export default fetchPlanetByIDWithBreaker;
