import { GraphQLObjectType, GraphQLString } from 'graphql';

const MeGraphQLType = new GraphQLObjectType({
  name: 'Me',
  fields: {
    id: { type: GraphQLString },
    name: { type: GraphQLString },
    firstName: { type: GraphQLString },
    lastName: { type: GraphQLString },
    bio: { type: GraphQLString },
  },
});

export default MeGraphQLType;
