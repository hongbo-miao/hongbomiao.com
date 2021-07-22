import { GraphQLInt, GraphQLObjectType } from 'graphql';

const FibonacciGraphQLType = new GraphQLObjectType({
  name: 'Fibonacci',
  fields: {
    n: { type: GraphQLInt },
    ans: { type: GraphQLInt },
  },
});

export default FibonacciGraphQLType;
