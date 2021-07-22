import DataLoader from 'dataloader';
import findUserByID from '../../dataSources/postgres/utils/findUserByID';
import formatUser from '../../dataSources/postgres/utils/formatUser';
import GraphQLUser from '../types/GraphQLUser';

const userDataLoader = (): DataLoader<string, GraphQLUser | null> =>
  new DataLoader(async (ids: ReadonlyArray<string>) => {
    return Promise.all(ids.map(async (id) => formatUser(await findUserByID(id))));
  });

export default userDataLoader;
