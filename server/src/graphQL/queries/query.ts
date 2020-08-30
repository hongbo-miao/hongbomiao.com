import { GraphQLInt, GraphQLNonNull, GraphQLObjectType } from 'graphql';
import FibonacciGraphQLType from '../graphQLTypes/Fibonacci.graphQLType';
import MeGraphQLType from '../graphQLTypes/Me.graphQLType';
import fibonacci from './fibonacci.query';
import me from './me.query';

const query = new GraphQLObjectType({
  name: 'Query',
  fields: {
    me: {
      type: MeGraphQLType,
      resolve: () => {
        return me();
      },
    },
    fibonacci: {
      type: FibonacciGraphQLType,
      args: {
        n: { type: new GraphQLNonNull(GraphQLInt) },
      },
      resolve: (parentValue, args) => {
        const { n } = args;
        return fibonacci(n);
      },
    },
  },
});

export default query;
