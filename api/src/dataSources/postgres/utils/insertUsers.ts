import PostgresInputUser from '../types/PostgresInputUser';
import PostgresUser from '../types/PostgresUser';
import insertUser from './insertUser';

const insertUsers = async (users: Array<PostgresInputUser>): Promise<PostgresUser[]> => {
  return Promise.all(users.map((user) => insertUser(user)));
};

export default insertUsers;
