import config from '../../config.js';
import findUserByEmail from '../../dataSources/postgres/utils/findUserByEmail.js';
import formatUser from '../../dataSources/postgres/utils/formatUser.js';
import GraphQLMe from '../types/GraphQLMe.js';

const getMe = async (): Promise<GraphQLMe | null> => {
  const { email } = config.seedUser;
  return formatUser(await findUserByEmail(email)) as GraphQLMe | null;
};

export default getMe;
