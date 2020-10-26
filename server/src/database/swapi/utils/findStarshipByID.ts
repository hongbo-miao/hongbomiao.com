import axios from 'axios';
import GraphQLStarship from '../../../graphQL/types/GraphQLStarship.type';
import formatStarship from './formatStarship';

const findStarshipByID = async (id: string): Promise<GraphQLStarship | null> => {
  const { data: swapiStarship } = await axios.get(`https://swapi.dev/api/starships/${id}/`);
  return formatStarship(id, swapiStarship);
};

export default findStarshipByID;
