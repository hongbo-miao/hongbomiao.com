import UserPostgresType from '../postgresTypes/user.postgresType';
import PostgresInputUser from '../types/PostgresInputUser.type';
import insertUser from './insertUser';

const insertUsers = async (users: Array<PostgresInputUser>): Promise<UserPostgresType[]> => {
  return Promise.all(users.map((user) => insertUser(user)));
};

export default insertUsers;
