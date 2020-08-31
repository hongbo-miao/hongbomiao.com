import findUserByID from '../../database/postgres/utils/findUserByID';
import GraphQLUser from '../types/GraphQLUser.type';

const getUser = async (id: string): Promise<GraphQLUser> => {
  const user = await findUserByID(id);
  if (user == null) {
    throw new Error('User does not exist.');
  }

  const { first_name: firstName, last_name: lastName, bio } = user;
  return {
    id,
    name: `${firstName} ${lastName}`,
    firstName,
    lastName,
    bio,
  };
};

export default getUser;
