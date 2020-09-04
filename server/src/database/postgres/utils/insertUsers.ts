import GraphQLUser from '../../../graphQL/types/GraphQLUser.type';
import PostgresInputUser from '../types/PostgresInputUser.type';
import insertUser from './insertUser';

const insertUsers = async (users: Array<PostgresInputUser>): Promise<(GraphQLUser | null)[]> => {
  return Promise.all(users.map((user) => insertUser(user)));
};

export default insertUsers;
