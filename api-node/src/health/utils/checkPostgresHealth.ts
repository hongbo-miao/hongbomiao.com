import pg from '../../dataSources/postgres/pg.js';

const checkPostgresHealth = async (): Promise<boolean> => {
  return pg.schema.hasTable('users');
};

export default checkPostgresHealth;
