import { GraphQLSchema } from 'graphql';
import query from '../queries/query';

const schema = new GraphQLSchema({
  query,
});

export default schema;
