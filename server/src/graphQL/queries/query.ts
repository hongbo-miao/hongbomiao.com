import { GraphQLID, GraphQLInt, GraphQLList, GraphQLNonNull, GraphQLObjectType } from 'graphql';
import FibonacciGraphQLType from '../graphQLTypes/Fibonacci.graphQLType';
import MeGraphQLType from '../graphQLTypes/Me.graphQLType';
import UserGraphQLType from '../graphQLTypes/User.graphQLType';
import getFibonacci from '../utils/getFibonacci';
import getMe from '../utils/getMe';

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
      resolve: async (parentValue, args, context) => {
        const { id } = args;
        const { dataLoaders } = context;
        return dataLoaders.user.load(id);
      },
    },
    users: {
      type: new GraphQLList(UserGraphQLType),
      args: {
        ids: { type: new GraphQLNonNull(new GraphQLList(new GraphQLNonNull(GraphQLID))) },
      },
      resolve: async (parentValue, args, context) => {
        const { ids } = args;
        const { dataLoaders } = context;
        return Promise.all(ids.map((id: string) => dataLoaders.user.load(id)));
      },
    },
  },
});

export default query;
