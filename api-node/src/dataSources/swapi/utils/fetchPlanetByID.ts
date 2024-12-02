import axios from 'axios';
import config from '../../../config.js';
import GraphQLPlanet from '../../../graphQL/types/GraphQLPlanet.js';
import formatPlanet from './formatPlanet.js';

const fetchPlanetByID = async (id: string): Promise<GraphQLPlanet | null> => {
  const { data: swapiPlanet } = await axios.get(`${config.swapiURL}/api/planets/${id}/`);
  return formatPlanet(id, swapiPlanet);
};

export default fetchPlanetByID;
