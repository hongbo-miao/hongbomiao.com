import GraphQLUser from '../types/GraphQLUser.type';
import getUser from './getUser';

const getUsers = async (ids: string[]): Promise<GraphQLUser[]> => {
  return Promise.all(ids.map((id) => getUser(id)));
};

export default getUsers;
