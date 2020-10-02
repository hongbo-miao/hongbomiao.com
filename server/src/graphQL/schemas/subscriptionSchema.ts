import { GraphQLSchema } from 'graphql';
import mutation from '../queries/mutation';
import query from '../queries/query';
import subscription from '../queries/subscription';

const subscriptionSchema = new GraphQLSchema({
  query,
  mutation,
  subscription,
});

export default subscriptionSchema;
