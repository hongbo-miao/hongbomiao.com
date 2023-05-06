import axios from 'axios';
import config from '../../../config';
import GraphQLPlanet from '../../../graphQL/types/GraphQLPlanet';
import formatPlanet from './formatPlanet';

const fetchPlanetByID = async (id: string): Promise<GraphQLPlanet | null> => {
  const { data: swapiPlanet } = await axios.get(`${config.swapiURL}/api/planets/${id}/`);
  return formatPlanet(id, swapiPlanet);
};

export default fetchPlanetByID;
