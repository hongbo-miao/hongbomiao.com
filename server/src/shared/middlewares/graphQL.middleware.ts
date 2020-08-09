import { graphqlHTTP } from 'express-graphql';
import schema from '../schemas/schema';

const graphQLMiddleware = graphqlHTTP({
  schema,
});

export default graphQLMiddleware;
