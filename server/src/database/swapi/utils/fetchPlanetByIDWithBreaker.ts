import axios from 'axios';
import config from '../../../config';
// eslint-disable-next-line import/no-cycle
import planetDataLoader from '../../../graphQL/dataLoaders/planet.dataLoader';
import GraphQLPlanet from '../../../graphQL/types/GraphQLPlanet.type';
import Breaker from '../../../reliability/utils/Breaker';
import formatPlanet from './formatPlanet';

const fetchPlanetByID = async (id: string): Promise<GraphQLPlanet | null> => {
  const { data: swapiPlanet } = await axios.get(`${config.swapiURL}/api/planets/${id}/`);
  return formatPlanet(id, swapiPlanet);
};

const breaker = new Breaker<GraphQLPlanet | null>('planet', fetchPlanetByID);

const fetchPlanetByIDWithBreaker = async (id: string): Promise<GraphQLPlanet | null> => {
  return breaker.fire(id, planetDataLoader);
};

export default fetchPlanetByIDWithBreaker;
