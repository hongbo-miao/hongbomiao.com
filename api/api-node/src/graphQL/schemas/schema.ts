import { GraphQLSchema } from 'graphql';
import mutation from '../queries/mutation.js';
import query from '../queries/query.js';

const schema = new GraphQLSchema({
  query,
  mutation,
});

export default schema;
