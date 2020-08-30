import { QueryBuilder } from 'knex';
import pg from '../pg';

const findUserByEmail = async (email: string): Promise<QueryBuilder> => {
  return pg.select('*').from('users').where('email', email.toLowerCase());
};

export default findUserByEmail;
