import GraphQLUser from '../../../graphQL/types/GraphQLUser.type';
import pg from '../pg';
import PostgresInputUser from '../types/PostgresInputUser.type';
import formatUser from './formatUser';

const insertUser = async (user: PostgresInputUser): Promise<GraphQLUser | null> => {
  const { email, password, firstName, lastName, bio } = user;

  const [firstUser] = await pg('users')
    .insert({
      email: email.toLowerCase(),
      password: pg.raw(`crypt('${password}', gen_salt('bf'))`),
      first_name: firstName,
      last_name: lastName,
      bio,
    })
    .returning('*');

  return formatUser(firstUser);
};

export default insertUser;
