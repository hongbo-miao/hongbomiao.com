import config from '../../config';
import findUserByEmail from '../../database/postgres/utils/findUserByEmail';
import formatUser from '../../database/postgres/utils/formatUser';
import GraphQLMe from '../types/GraphQLMe.type';

const getMe = async (): Promise<GraphQLMe | null> => {
  const { email } = config.seedUser;
  return formatUser(await findUserByEmail(email)) as GraphQLMe | null;
};

export default getMe;
