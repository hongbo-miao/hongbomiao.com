import pg from '../pg.js';
import PostgresUser from '../types/PostgresUser.js';

const findUserByID = async (id: string): Promise<PostgresUser> => {
  const [firstUser] = await pg.select('*').from('users').where('id', id);
  return firstUser;
};

export default findUserByID;
