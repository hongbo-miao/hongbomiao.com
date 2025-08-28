import config from '../../../config.js';
import GraphQLPlanet from '../../../graphQL/types/GraphQLPlanet.js';
import formatPlanet from './formatPlanet.js';

const fetchPlanetById = async (id: string): Promise<GraphQLPlanet | null> => {
  const response = await fetch(`${config.swapiUrl}/api/planets/${id}/`);
  if (!response.ok) {
    throw new Error(`Failed to fetch planet ${id}: ${response.status} ${response.statusText}`);
  }
  const swapiPlanet = await response.json();
  return formatPlanet(id, swapiPlanet);
};

export default fetchPlanetById;
