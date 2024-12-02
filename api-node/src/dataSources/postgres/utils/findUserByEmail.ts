import pg from '../pg.js';
import PostgresUser from '../types/PostgresUser.js';

const findUserByEmail = async (email: string): Promise<PostgresUser> => {
  const [firstUser] = await pg.select('*').from('users').where('email', email.toLowerCase());
  return firstUser;
};

export default findUserByEmail;
