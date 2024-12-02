import DataLoader from 'dataloader';
import fetchPlanetByID from '../../dataSources/swapi/utils/fetchPlanetByID.js';
import GraphQLPlanet from '../types/GraphQLPlanet.js';

const planetDataLoader = (): DataLoader<string, GraphQLPlanet | null> =>
  new DataLoader(async (ids: ReadonlyArray<string>) => {
    return Promise.all(ids.map((id) => fetchPlanetByID(id)));
  });

export default planetDataLoader;
