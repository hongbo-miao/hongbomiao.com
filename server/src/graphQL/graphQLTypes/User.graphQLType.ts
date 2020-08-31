import { GraphQLObjectType, GraphQLString } from 'graphql';

const UserGraphQLType = new GraphQLObjectType({
  name: 'User',
  fields: {
    id: { type: GraphQLString },
    name: { type: GraphQLString },
    firstName: { type: GraphQLString },
    lastName: { type: GraphQLString },
    bio: { type: GraphQLString },
  },
});

export default UserGraphQLType;
