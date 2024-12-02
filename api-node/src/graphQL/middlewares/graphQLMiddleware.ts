import { graphqlHTTP } from 'express-graphql';
import { NoSchemaIntrospectionCustomRule } from 'graphql';
import depthLimit from 'graphql-depth-limit';
import { applyMiddleware } from 'graphql-middleware';
import logger from '../../log/utils/logger.js';
import verifyJWTTokenAndExtractMyID from '../../security/utils/verifyJWTTokenAndExtractMyID.js';
import isProduction from '../../shared/utils/isProduction.js';
import dataLoaders from '../dataLoaders/dataLoaders.js';
import permissions from '../permissions/permissions.js';
import schema from '../schemas/schema.js';

const graphQLMiddleware = graphqlHTTP(req => {
  return {
    context: {
      dataLoaders: dataLoaders(),
      myId: verifyJWTTokenAndExtractMyID(req.headers.authorization),
    },
    customFormatErrorFn: (err) => {
      const stackErr = {
        ...err,
        stack: err.stack ? err.stack.split('\n') : [],
      };
      logger.error(stackErr, 'graphQLMiddleware');
      return isProduction() ? err : stackErr;
    },
    schema: applyMiddleware(schema, permissions),
    validationRules: [
      ...(isProduction() ? [NoSchemaIntrospectionCustomRule] : []),
      depthLimit(5),
    ],
  };
});

export default graphQLMiddleware;
