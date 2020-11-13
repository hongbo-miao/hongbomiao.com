import DataLoader from 'dataloader';
import findUserByID from '../../database/postgres/utils/findUserByID';
import formatUser from '../../database/postgres/utils/formatUser';
import GraphQLUser from '../types/GraphQLUser.type';

const batchGetUsers = async (ids: ReadonlyArray<string>): Promise<(GraphQLUser | null)[]> => {
  return Promise.all(ids.map(async (id) => formatUser(await findUserByID(id))));
};

const userDataLoader = new DataLoader(batchGetUsers);

export default userDataLoader;
