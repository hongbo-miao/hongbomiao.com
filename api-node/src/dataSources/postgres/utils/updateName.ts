import pg from '../pg.js';
import PostgresUser from '../types/PostgresUser.js';

const updateName = async (id: string, firstName: string, lastName: string): Promise<PostgresUser> => {
  const [firstUser] = await pg('users')
    .where({ id })
    .update({
      first_name: firstName,
      last_name: lastName,
    })
    .returning('*');

  if (firstUser == null) {
    throw new Error('User does not exist.');
  }
  return firstUser;
};

export default updateName;
