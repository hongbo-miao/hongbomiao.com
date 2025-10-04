import DataLoader from 'dataloader';
import fetchStarshipById from '../../dataSources/swapi/utils/fetchStarshipById.js';
import GraphQLStarship from '../types/GraphQLStarship.js';

const starshipDataLoader = (): DataLoader<string, GraphQLStarship | null> =>
  new DataLoader(async (ids: ReadonlyArray<string>) => {
    return Promise.all(ids.map((id) => fetchStarshipById(id)));
  });

export default starshipDataLoader;
