import { GraphQLObjectType, GraphQLString } from 'graphql';

const MeType = new GraphQLObjectType({
  name: 'Me',
  fields: {
    name: { type: GraphQLString },
    slogan: { type: GraphQLString },
  },
});

export default MeType;
