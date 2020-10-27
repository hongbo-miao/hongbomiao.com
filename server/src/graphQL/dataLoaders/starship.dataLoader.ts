import DataLoader from 'dataloader';
// eslint-disable-next-line import/no-cycle
import fetchStarshipByIDWithBreaker from '../../database/swapi/utils/fetchStarshipByIDWithBreaker';
import GraphQLStarship from '../types/GraphQLStarship.type';

const batchGetStarships = async (ids: ReadonlyArray<string>): Promise<(GraphQLStarship | null)[]> => {
  return Promise.all(ids.map((id) => fetchStarshipByIDWithBreaker(id)));
};

const starshipDataLoader = new DataLoader(batchGetStarships);

export default starshipDataLoader;
