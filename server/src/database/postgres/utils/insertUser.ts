import { QueryBuilder } from 'knex';
import pg from '../pg';

const insertUser = async (
  email: string,
  password: string,
  firstName: string,
  lastName: string
): Promise<QueryBuilder> => {
  return pg('users')
    .returning('id')
    .insert({
      email: email.toLowerCase(),
      password: pg.raw(`crypt('${password}', gen_salt('bf'))`),
      first_name: firstName,
      last_name: lastName,
    });
};

export default insertUser;
