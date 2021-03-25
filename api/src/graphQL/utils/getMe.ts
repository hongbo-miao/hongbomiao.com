import config from '../../config';
import findUserByEmail from '../../dataSources/postgres/utils/findUserByEmail';
import formatUser from '../../dataSources/postgres/utils/formatUser';
import GraphQLMe from '../types/GraphQLMe';

const getMe = async (): Promise<GraphQLMe | null> => {
  const { email } = config.seedUser;
  return formatUser(await findUserByEmail(email)) as GraphQLMe | null;
};

export default getMe;
