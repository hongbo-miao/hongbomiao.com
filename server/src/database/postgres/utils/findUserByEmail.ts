import pg from '../pg';
import PostgresUser from '../types/PostgresUser.type';

const findUserByEmail = async (email: string): Promise<PostgresUser> => {
  const [firstUser] = await pg.select('*').from('users').where('email', email.toLowerCase());
  return firstUser;
};

export default findUserByEmail;
