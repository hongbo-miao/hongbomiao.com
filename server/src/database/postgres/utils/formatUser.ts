import GraphQLUser from '../../../graphQL/types/GraphQLUser.type';
import PostgresUserType from '../types/PostgresUser.type';

const formatUser = (user: PostgresUserType): GraphQLUser | null => {
  if (user == null) return null;

  const { id, first_name: firstName, last_name: lastName, bio } = user;
  return {
    id,
    name: `${firstName} ${lastName}`,
    firstName,
    lastName,
    bio,
  };
};

export default formatUser;
