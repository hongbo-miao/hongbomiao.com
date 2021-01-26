import { GraphQLSchema } from 'graphql';
import mutation from '../queries/mutation';
import query from '../queries/query';

const schema = new GraphQLSchema({
  query,
  mutation,
});

export default schema;
