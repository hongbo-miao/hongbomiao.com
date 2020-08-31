import Config from '../../Config';
import findUserByEmail from '../../database/postgres/utils/findUserByEmail';
import GraphQLMe from '../types/GraphQLMe.type';

const getMe = async (): Promise<GraphQLMe> => {
  const { email } = Config.seedUser;
  if (email == null) {
    throw new Error('Missing seed user.');
  }

  const user = await findUserByEmail(email);
  if (user == null) {
    throw new Error('User does not exist.');
  }

  const { id, first_name: firstName, last_name: lastName, bio } = user;
  return {
    id,
    name: `${firstName} ${lastName}`,
    firstName,
    lastName,
    bio,
  };
};

export default getMe;
