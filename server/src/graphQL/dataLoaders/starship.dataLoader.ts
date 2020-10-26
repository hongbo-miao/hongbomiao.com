import DataLoader from 'dataloader';
import findStarshipByID from '../../database/swapi/utils/findStarshipByID';
import GraphQLStarship from '../types/GraphQLStarship.type';

const batchGetStarships = async (ids: ReadonlyArray<string>): Promise<(GraphQLStarship | null)[]> => {
  return Promise.all(ids.map((id) => findStarshipByID(id)));
};

const starshipDataLoader = new DataLoader(batchGetStarships);

export default starshipDataLoader;
