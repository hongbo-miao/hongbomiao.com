import DataLoader from 'dataloader';
import fetchStarshipByID from '../../dataSources/swapi/utils/fetchStarshipByID';
import GraphQLStarship from '../types/GraphQLStarship';

const starshipDataLoader = (): DataLoader<string, GraphQLStarship | null> =>
  new DataLoader(async (ids: ReadonlyArray<string>) => {
    return Promise.all(ids.map((id) => fetchStarshipByID(id)));
  });

export default starshipDataLoader;
