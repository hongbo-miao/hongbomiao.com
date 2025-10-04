import { GraphQLID, GraphQLObjectType, GraphQLString } from 'graphql';

const MeGraphQLType = new GraphQLObjectType({
  name: 'Me',
  fields: {
    id: { type: GraphQLID },
    name: { type: GraphQLString },
    firstName: { type: GraphQLString },
    lastName: { type: GraphQLString },
    email: { type: GraphQLString },
    bio: { type: GraphQLString },
  },
});

export default MeGraphQLType;
