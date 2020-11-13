import GraphQLMe from '../../../graphQL/types/GraphQLMe.type';
import GraphQLUser from '../../../graphQL/types/GraphQLUser.type';
import PostgresUser from '../types/PostgresUser.type';

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
