import DataLoader from 'dataloader';
import GraphQLUser from '../types/GraphQLUser.type';
import getUser from '../utils/getUser';

const batchGetUsers = async (ids: readonly string[]): Promise<GraphQLUser[]> => {
  return Promise.all(ids.map((id) => getUser(id)));
};

const userDataLoader = new DataLoader(batchGetUsers);

export default userDataLoader;
