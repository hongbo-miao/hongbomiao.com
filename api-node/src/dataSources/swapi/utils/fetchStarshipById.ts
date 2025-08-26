import config from '../../../config.js';
import GraphQLStarship from '../../../graphQL/types/GraphQLStarship.js';
import formatStarship from './formatStarship.js';

const fetchStarshipById = async (id: string): Promise<GraphQLStarship | null> => {
  const response = await fetch(`${config.swapiURL}/api/starships/${id}/`);
  if (!response.ok) {
    throw new Error(`Failed to fetch starship ${id}: ${response.status} ${response.statusText}`);
  }
  const swapiStarship = await response.json();
  return formatStarship(id, swapiStarship);
};

export default fetchStarshipById;
