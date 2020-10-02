import config from '../../config';
import findUserByEmail from '../../database/postgres/utils/findUserByEmail';
import GraphQLMe from '../types/GraphQLMe.type';

const getMe = async (): Promise<GraphQLMe | null> => {
  const { email } = config.seedUser;
  return findUserByEmail(email);
};

export default getMe;
