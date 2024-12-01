import { GraphQLSchema } from 'graphql';
import mutation from '../queries/mutation.js';
import query from '../queries/query.js';
import subscription from '../queries/subscription.js';

const subscriptionSchema = new GraphQLSchema({
  query,
  mutation,
  subscription,
});

export default subscriptionSchema;
