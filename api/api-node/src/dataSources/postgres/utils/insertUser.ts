import bcrypt from 'bcrypt';
import pg from '../pg.js';
import PostgresInputUser from '../types/PostgresInputUser.js';
import PostgresUser from '../types/PostgresUser.js';

const insertUser = async (user: PostgresInputUser): Promise<PostgresUser> => {
  const { email, password, firstName, lastName, bio } = user;

  const [firstUser] = await pg('users')
    .insert({
      email: email.toLowerCase(),
      password: await bcrypt.hash(password, 10),
      first_name: firstName,
      last_name: lastName,
      bio,
    })
    .returning('*');

  return firstUser;
};

export default insertUser;
