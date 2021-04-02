import DataLoader from 'dataloader';
import fetchPlanetByIDWithBreaker from '../../dataSources/swapi/utils/fetchPlanetByIDWithBreaker';
import GraphQLPlanet from '../types/GraphQLPlanet';

const planetDataLoader = (): DataLoader<string, GraphQLPlanet | null> =>
  new DataLoader(async (ids: ReadonlyArray<string>) => {
    return Promise.all(ids.map((id) => fetchPlanetByIDWithBreaker(id)));
  });

export default planetDataLoader;
