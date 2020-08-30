import { GraphQLObjectType, GraphQLString } from 'graphql';

const MeGraphQLType = new GraphQLObjectType({
  name: 'Me',
  fields: {
    name: { type: GraphQLString },
    bio: { type: GraphQLString },
  },
});

export default MeGraphQLType;
