import { graphqlHTTP } from 'express-graphql';
import depthLimit from 'graphql-depth-limit';
import { applyMiddleware } from 'graphql-middleware';
import verifyJWTToken from '../../security/utils/verifyJWTToken';
import isDevelopment from '../../shared/utils/isDevelopment';
import planetDataLoader from '../dataLoaders/planet.dataLoader';
import starshipDataLoader from '../dataLoaders/starship.dataLoader';
import userDataLoader from '../dataLoaders/user.dataLoader';
import permissions from '../permissions/permissions';
import schema from '../schemas/schema';

const graphQLMiddleware = graphqlHTTP((req) => ({
  context: {
    dataLoaders: {
      planet: planetDataLoader,
      starship: starshipDataLoader,
      user: userDataLoader,
    },
    myId: verifyJWTToken(req.headers.authorization),
  },
  customFormatErrorFn: (err) => {
    if (isDevelopment()) {
      return {
        ...err,
        stack: err.stack ? err.stack.split('\n') : [],
      };
    }
    return err;
  },
  schema: applyMiddleware(schema, permissions),
  validationRules: [depthLimit(5)],
}));

export default graphQLMiddleware;
