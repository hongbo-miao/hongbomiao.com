import { graphqlHTTP } from 'express-graphql';
import userDataLoader from '../dataLoaders/user.dataLoader';
import schema from '../schemas/schema';

const graphQLMiddleware = graphqlHTTP({
  context: {
    dataLoaders: {
      user: userDataLoader,
    },
  },
  schema,
});

export default graphQLMiddleware;
