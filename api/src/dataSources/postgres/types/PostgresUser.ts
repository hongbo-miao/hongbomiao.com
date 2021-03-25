/* eslint-disable camelcase */

type PostgresUser = {
  id: string;
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  bio: string | null;
  created_on: Date;
  last_login: Date | null;
};

export default PostgresUser;
