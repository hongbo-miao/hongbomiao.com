import pg from '../pg';
import PostgresUserType from '../types/PostgresUser.type';

const findUserByID = async (id: string): Promise<PostgresUserType | undefined> => {
  return pg.select('*').from('users').where('id', id).first();
};

export default findUserByID;
