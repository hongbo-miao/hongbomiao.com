import PostgresInputUser from '../types/PostgresInputUser.type';
import PostgresUser from '../types/PostgresUser.type';
import insertUser from './insertUser';

const insertUsers = async (users: Array<PostgresInputUser>): Promise<PostgresUser[]> => {
  return Promise.all(users.map((user) => insertUser(user)));
};

export default insertUsers;
