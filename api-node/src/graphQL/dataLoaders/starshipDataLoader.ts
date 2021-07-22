import DataLoader from 'dataloader';
import fetchStarshipByIDWithBreaker from '../../dataSources/swapi/utils/fetchStarshipByIDWithBreaker';
import GraphQLStarship from '../types/GraphQLStarship';

const starshipDataLoader = (): DataLoader<string, GraphQLStarship | null> =>
  new DataLoader(async (ids: ReadonlyArray<string>) => {
    return Promise.all(ids.map((id) => fetchStarshipByIDWithBreaker(id)));
  });

export default starshipDataLoader;
