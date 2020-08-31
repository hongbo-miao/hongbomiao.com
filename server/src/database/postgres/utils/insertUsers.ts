import PostgresInputUser from '../types/PostgresInputUser.type';
import PostgresUserType from '../types/PostgresUser.type';
import insertUser from './insertUser';

const insertUsers = async (users: Array<PostgresInputUser>): Promise<PostgresUserType[]> => {
  return Promise.all(users.map((user) => insertUser(user)));
};

export default insertUsers;
