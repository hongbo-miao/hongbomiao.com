import axios from 'axios';
import config from '../../../config';
import GraphQLStarship from '../../../graphQL/types/GraphQLStarship';
import createCircuitBreaker from '../../../reliability/utils/createCircuitBreaker';
import formatStarship from './formatStarship';

const fetchStarshipByID = async (id: string): Promise<GraphQLStarship | null> => {
  const { data: swapiStarship } = await axios.get(`${config.swapiURL}/api/starships/${id}/`);
  return formatStarship(id, swapiStarship);
};

const breaker = createCircuitBreaker(fetchStarshipByID);

const fetchStarshipByIDWithBreaker = async (id: string): Promise<GraphQLStarship | null> => {
  return breaker.fire(id);
};

export default fetchStarshipByIDWithBreaker;
