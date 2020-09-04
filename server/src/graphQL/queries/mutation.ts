import { GraphQLID, GraphQLNonNull, GraphQLObjectType, GraphQLString } from 'graphql';
import updateName from '../../database/postgres/utils/updateName';
import UserGraphQLType from '../graphQLTypes/User.graphQLType';

const mutation = new GraphQLObjectType({
  name: 'Mutation',
  fields: {
    updateName: {
      type: UserGraphQLType,
      args: {
        id: { type: new GraphQLNonNull(GraphQLID) },
        firstName: { type: new GraphQLNonNull(GraphQLString) },
        lastName: { type: new GraphQLNonNull(GraphQLString) },
      },
      resolve: async (parentValue, args) => {
        const { id, firstName, lastName } = args;
        return updateName(id, firstName, lastName);
      },
    },
  },
});

export default mutation;
