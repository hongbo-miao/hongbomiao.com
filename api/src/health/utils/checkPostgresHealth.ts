import pg from '../../database/postgres/pg';

const checkPostgresHealth = async (): Promise<boolean> => {
  const hasUsers = await pg.schema.hasTable('users');
  return hasUsers != null;
};

export default checkPostgresHealth;
