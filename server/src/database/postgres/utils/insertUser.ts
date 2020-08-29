import { QueryBuilder } from 'knex';
import pg from '../pg';

const insertUser = async (
  email: string,
  password: string,
  firstname: string,
  lastname: string
): Promise<QueryBuilder> => {
  return pg('users').insert({
    email: email.toLowerCase(),
    password: pg.raw(`crypt('${password}', gen_salt('bf'))`),
    firstname,
    lastname,
  });
};

export default insertUser;
