import pg from '../pg';
import UserPostgresType from '../postgresTypes/user.postgresType';

const insertUser = async (
  email: string,
  password: string,
  firstName: string,
  lastName: string,
  bio: string | null | undefined
): Promise<UserPostgresType> => {
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
