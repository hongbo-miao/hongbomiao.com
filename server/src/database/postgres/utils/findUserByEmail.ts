import pg from '../pg';
import PostgresUserType from '../types/PostgresUser.type';

const findUserByEmail = async (email: string): Promise<PostgresUserType | undefined> => {
  return pg.select('*').from('users').where('email', email.toLowerCase()).first();
};

export default findUserByEmail;
