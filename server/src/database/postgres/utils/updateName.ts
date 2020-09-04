import GraphQLUser from '../../../graphQL/types/GraphQLUser.type';
import pg from '../pg';
import formatUser from './formatUser';

const updateName = async (id: string, firstName: string, lastName: string): Promise<GraphQLUser | null> => {
  const [firstUser] = await pg('users')
    .where({ id })
    .update({
      first_name: firstName,
      last_name: lastName,
    })
    .returning('*');

  if (firstUser == null) {
    throw new Error('User does not exist.');
  }
  return formatUser(firstUser);
};

export default updateName;
