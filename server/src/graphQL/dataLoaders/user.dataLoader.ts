import DataLoader from 'dataloader';
import findUserByID from '../../database/postgres/utils/findUserByID';
import GraphQLUser from '../types/GraphQLUser.type';

const batchGetUsers = async (ids: ReadonlyArray<string>): Promise<(GraphQLUser | null)[]> => {
  return Promise.all(ids.map((id) => findUserByID(id)));
};

const userDataLoader = new DataLoader(batchGetUsers);

export default userDataLoader;
