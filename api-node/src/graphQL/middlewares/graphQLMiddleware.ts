import { graphqlHTTP } from 'express-graphql';
import { NoSchemaIntrospectionCustomRule } from 'graphql';
import depthLimit from 'graphql-depth-limit';
import { applyMiddleware } from 'graphql-middleware';
import queryComplexity, { simpleEstimator } from 'graphql-query-complexity';
import logger from '../../log/utils/logger';
import verifyJWTTokenAndExtractMyID from '../../security/utils/verifyJWTTokenAndExtractMyID';
import isProduction from '../../shared/utils/isProduction';
import dataLoaders from '../dataLoaders/dataLoaders';
import permissions from '../permissions/permissions';
import schema from '../schemas/schema';

const graphQLMiddleware = graphqlHTTP((req, res, params) => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  const { variables } = params;
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
      queryComplexity({
        estimators: [simpleEstimator({ defaultComplexity: 1 })],
        maximumComplexity: 1000,
        variables,
        onComplete: (complexity: number) => {
          logger.info({ complexity }, 'queryComplexity');
        },
      }),
    ],
  };
});

export default graphQLMiddleware;
