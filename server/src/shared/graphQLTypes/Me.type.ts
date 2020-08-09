import { GraphQLObjectType, GraphQLString } from 'graphql';

const MeGraphQLType = new GraphQLObjectType({
  name: 'Me',
  fields: {
    name: { type: GraphQLString },
    slogan: { type: GraphQLString },
  },
});

export default MeGraphQLType;
