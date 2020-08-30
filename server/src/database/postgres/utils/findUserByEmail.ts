import pg from '../pg';
import UserPostgresType from '../postgresTypes/user.postgresType';

const findUserByEmail = async (email: string): Promise<UserPostgresType | undefined> => {
  return pg.select('*').from('users').where('email', email.toLowerCase()).first();
};

export default findUserByEmail;
