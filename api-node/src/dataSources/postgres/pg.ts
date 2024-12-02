import knex from 'knex';
import config from '../../config.js';

const pg = knex({
  client: 'pg',
  connection: config.postgresConnection,
  pool: {
    min: 2,
    max: 10,
  },
});

export default pg;
