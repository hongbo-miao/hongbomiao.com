import pg from '../pg';
import PostgresUser from '../types/PostgresUser';

const findUserByID = async (id: string): Promise<PostgresUser> => {
  const [firstUser] = await pg.select('*').from('users').where('id', id);
  return firstUser;
};

export default findUserByID;
