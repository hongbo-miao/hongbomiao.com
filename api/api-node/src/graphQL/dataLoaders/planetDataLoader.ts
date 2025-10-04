import DataLoader from 'dataloader';
import fetchPlanetById from '../../dataSources/swapi/utils/fetchPlanetById.js';
import GraphQLPlanet from '../types/GraphQLPlanet.js';

const planetDataLoader = (): DataLoader<string, GraphQLPlanet | null> =>
  new DataLoader(async (ids: ReadonlyArray<string>) => {
    return Promise.all(ids.map((id) => fetchPlanetById(id)));
  });

export default planetDataLoader;
