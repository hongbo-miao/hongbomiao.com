import axios from 'axios';
import config from '../../../config';
import GraphQLStarship from '../../../graphQL/types/GraphQLStarship';
import formatStarship from './formatStarship';

const fetchStarshipByID = async (id: string): Promise<GraphQLStarship | null> => {
  const { data: swapiStarship } = await axios.get(`${config.swapiURL}/api/starships/${id}/`);
  return formatStarship(id, swapiStarship);
};

export default fetchStarshipByID;
