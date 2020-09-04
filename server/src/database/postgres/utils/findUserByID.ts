import GraphQLUser from '../../../graphQL/types/GraphQLUser.type';
import pg from '../pg';
import formatUser from './formatUser';

const findUserByID = async (id: string): Promise<GraphQLUser | null> => {
  const [firstUser] = await pg.select('*').from('users').where('id', id);
  return formatUser(firstUser);
};

export default findUserByID;
