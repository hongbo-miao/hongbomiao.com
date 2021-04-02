import { graphqlHTTP } from 'express-graphql';
import depthLimit from 'graphql-depth-limit';
import { applyMiddleware } from 'graphql-middleware';
import queryComplexity, { simpleEstimator } from 'graphql-query-complexity';
import logger from '../../log/utils/logger';
import verifyJWTToken from '../../security/utils/verifyJWTToken';
import isDevelopment from '../../shared/utils/isDevelopment';
import dataLoaders from '../dataLoaders/dataLoaders';
import permissions from '../permissions/permissions';
import schema from '../schemas/schema';

const graphQLMiddleware = graphqlHTTP((req, res, params) => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  const { variables } = params;
  return {
    context: {
      dataLoaders: dataLoaders(),
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
    validationRules: [
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
