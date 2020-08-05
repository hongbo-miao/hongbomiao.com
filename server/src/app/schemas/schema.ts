import { GraphQLSchema } from 'graphql';
import Query from '../queries/query';

const schema = new GraphQLSchema({
  query: Query,
});

export default schema;
