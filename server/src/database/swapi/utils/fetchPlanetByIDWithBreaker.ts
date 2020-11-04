import axios from 'axios';
import config from '../../../config';
// eslint-disable-next-line import/no-cycle
import planetDataLoader from '../../../graphQL/dataLoaders/planet.dataLoader';
import GraphQLPlanet from '../../../graphQL/types/GraphQLPlanet.type';
import createCircuitBreaker from '../../../reliability/utils/createCircuitBreaker';
import formatPlanet from './formatPlanet';

const fetchPlanetByID = async (id: string): Promise<GraphQLPlanet | null> => {
  const { data: swapiPlanet } = await axios.get(`${config.swapiURL}/api/planets/${id}/`);
  return formatPlanet(id, swapiPlanet);
};

const breaker = createCircuitBreaker(fetchPlanetByID);

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
