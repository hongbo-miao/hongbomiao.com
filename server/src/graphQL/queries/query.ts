import { GraphQLID, GraphQLInt, GraphQLList, GraphQLNonNull, GraphQLObjectType } from 'graphql';
import FibonacciGraphQLType from '../graphQLTypes/Fibonacci.graphQLType';
import MeGraphQLType from '../graphQLTypes/Me.graphQLType';
import UserGraphQLType from '../graphQLTypes/User.graphQLType';
import getFibonacci from '../utils/getFibonacci';
import getMe from '../utils/getMe';
import getUser from '../utils/getUser';
import getUsers from '../utils/getUsers';

const query = new GraphQLObjectType({
  name: 'Query',
  fields: {
    me: {
      type: MeGraphQLType,
      resolve: async () => {
        return getMe();
      },
    },
    fibonacci: {
      type: FibonacciGraphQLType,
      args: {
        n: { type: new GraphQLNonNull(GraphQLInt) },
      },
      resolve: (parentValue, args) => {
        const { n } = args;
        return getFibonacci(n);
      },
    },
    user: {
      type: UserGraphQLType,
      args: {
        id: { type: new GraphQLNonNull(GraphQLID) },
      },
      resolve: async (parentValue, args) => {
        const { id } = args;
        return getUser(id);
      },
    },
    users: {
      type: new GraphQLList(UserGraphQLType),
      args: {
        ids: { type: new GraphQLNonNull(new GraphQLList(new GraphQLNonNull(GraphQLID))) },
      },
      resolve: async (parentValue, args) => {
        const { ids } = args;
        return getUsers(ids);
      },
    },
  },
});

export default query;
