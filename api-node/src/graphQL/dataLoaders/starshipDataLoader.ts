import DataLoader from 'dataloader';
import fetchStarshipByID from '../../dataSources/swapi/utils/fetchStarshipByID.js';
import GraphQLStarship from '../types/GraphQLStarship.js';

const starshipDataLoader = (): DataLoader<string, GraphQLStarship | null> =>
  new DataLoader(async (ids: ReadonlyArray<string>) => {
    return Promise.all(ids.map((id) => fetchStarshipByID(id)));
  });

export default starshipDataLoader;
