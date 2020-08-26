import { GraphQLInt, GraphQLNonNull, GraphQLObjectType } from 'graphql';
import FibonacciGraphQLType from '../graphQLTypes/Fibonacci.type';
import MeGraphQLType from '../graphQLTypes/Me.type';
import calFibonacci from '../utils/calFibonacci';

const query = new GraphQLObjectType({
  name: 'Query',
  fields: {
    me: {
      type: MeGraphQLType,
      resolve: () => {
        return { name: 'Hongbo Miao', slogan: 'Making magic happen' };
      },
    },
    fibonacci: {
      type: FibonacciGraphQLType,
      args: {
        n: { type: new GraphQLNonNull(GraphQLInt) },
      },
      resolve: (parentValue, args) => {
        const { n } = args;
        return {
          n,
          ans: calFibonacci(n),
        };
      },
    },
  },
});

export default query;
