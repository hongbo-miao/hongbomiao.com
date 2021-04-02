import { rule, shield } from 'graphql-shield';
import isDevelopment from '../../shared/utils/isDevelopment';

const isAuthenticated = rule()(async (parentValue, args, context) => {
  const { myId } = context;
  return myId != null;
});

const permissions = shield(
  {
    Query: {
      fibonacci: isAuthenticated,
      user: isAuthenticated,
      users: isAuthenticated,
      planet: isAuthenticated,
      planets: isAuthenticated,
      starship: isAuthenticated,
      starships: isAuthenticated,
    },
    Mutation: {
      updateName: isAuthenticated,
      uploadFile: isAuthenticated,
    },
  },
  { debug: isDevelopment() },
);

export default permissions;
