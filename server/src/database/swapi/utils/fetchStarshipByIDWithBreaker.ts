import axios from 'axios';
import config from '../../../config';
// eslint-disable-next-line import/no-cycle
import starshipDataLoader from '../../../graphQL/dataLoaders/starship.dataLoader';
import GraphQLStarship from '../../../graphQL/types/GraphQLStarship.type';
import Breaker from '../../../reliability/utils/Breaker';
import formatStarship from './formatStarship';

const fetchStarshipByID = async (id: string): Promise<GraphQLStarship | null> => {
  const { data: swapiStarship } = await axios.get(`${config.swapiURL}/api/starships/${id}/`);
  return formatStarship(id, swapiStarship);
};

const breaker = new Breaker<GraphQLStarship | null>('starship', fetchStarshipByID);

const fetchStarshipByIDWithBreaker = async (id: string): Promise<GraphQLStarship | null> => {
  return breaker.fire(id, starshipDataLoader);
};

export default fetchStarshipByIDWithBreaker;
