import pg from '../../dataSources/postgres/pg';

const checkPostgresHealth = async (): Promise<boolean> => {
  return pg.schema.hasTable('users');
};

export default checkPostgresHealth;
