import GraphQLUser from '../../../graphQL/types/GraphQLUser.type';
import pg from '../pg';
import formatUser from './formatUser';

const findUserByEmail = async (email: string): Promise<GraphQLUser | null> => {
  const [firstUser] = await pg.select('*').from('users').where('email', email.toLowerCase());
  return formatUser(firstUser);
};

export default findUserByEmail;
