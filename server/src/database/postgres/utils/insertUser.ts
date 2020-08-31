import pg from '../pg';
import PostgresInputUser from '../types/PostgresInputUser.type';
import PostgresUserType from '../types/PostgresUser.type';

const insertUser = async (user: PostgresInputUser): Promise<PostgresUserType> => {
  const { email, password, firstName, lastName, bio } = user;
  return pg('users')
    .returning('*')
    .insert({
      email: email.toLowerCase(),
      password: pg.raw(`crypt('${password}', gen_salt('bf'))`),
      first_name: firstName,
      last_name: lastName,
      bio,
    });
};

export default insertUser;
