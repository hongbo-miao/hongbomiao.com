import Config from '../../Config';
import findUserByEmail from '../../database/postgres/utils/findUserByEmail';
import GraphQLMe from '../types/GraphQLMe.type';

const getMe = async (): Promise<GraphQLMe | null> => {
  const { email } = Config.seedUser;
  if (email == null) {
    throw new Error('Missing seed user.');
  }
  return findUserByEmail(email);
};

export default getMe;
