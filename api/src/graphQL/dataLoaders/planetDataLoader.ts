import DataLoader from 'dataloader';
// eslint-disable-next-line import/no-cycle
import fetchPlanetByIDWithBreaker from '../../database/swapi/utils/fetchPlanetByIDWithBreaker';
import GraphQLPlanet from '../types/GraphQLPlanet.type';

const batchGetPlanets = async (ids: ReadonlyArray<string>): Promise<(GraphQLPlanet | null)[]> => {
  return Promise.all(ids.map((id) => fetchPlanetByIDWithBreaker(id)));
};

const planetDataLoader = new DataLoader(batchGetPlanets);

export default planetDataLoader;
