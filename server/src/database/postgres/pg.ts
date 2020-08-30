import knex from 'knex';
import Config from '../../Config';

const pg = knex({
  client: 'pg',
  connection: Config.postgresConnection,
  pool: {
    min: 2,
    max: 10,
  },
});

export default pg;
