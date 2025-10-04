import PostgresInputUser from '../types/PostgresInputUser.js';
import PostgresUser from '../types/PostgresUser.js';
import insertUser from './insertUser.js';

const insertUsers = async (users: Array<PostgresInputUser>): Promise<PostgresUser[]> => {
  return Promise.all(users.map((user) => insertUser(user)));
};

export default insertUsers;
