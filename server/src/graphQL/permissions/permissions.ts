import { rule, shield } from 'graphql-shield';

const isAuthenticated = rule()(async (parentValue, args, context) => {
  const { myId } = context;
  return myId != null;
});

const permissions = shield({
  Query: {
    user: isAuthenticated,
    users: isAuthenticated,
    planet: isAuthenticated,
    planets: isAuthenticated,
    starship: isAuthenticated,
    starships: isAuthenticated,
  },
  Mutation: {
    updateName: isAuthenticated,
  },
});

export default permissions;
