import GraphQLMe from '../../../graphQL/types/GraphQLMe.js';
import GraphQLUser from '../../../graphQL/types/GraphQLUser.js';
import PostgresUser from '../types/PostgresUser.js';

const formatUser = (user: PostgresUser): GraphQLMe | GraphQLUser | null => {
  if (user == null) return null;

  const { id, first_name: firstName, last_name: lastName, email, bio } = user;
  return {
    id,
    name: `${firstName} ${lastName}`,
    firstName,
    lastName,
    email,
    bio,
  };
};

export default formatUser;
