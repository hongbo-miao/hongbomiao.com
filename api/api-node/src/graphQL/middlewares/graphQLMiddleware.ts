import type { Request } from 'express';
import { NoSchemaIntrospectionCustomRule } from 'graphql';
import depthLimit from 'graphql-depth-limit';
import { createHandler } from 'graphql-http/lib/use/express';
import { applyMiddleware } from 'graphql-middleware';
import verifyJWTTokenAndExtractMyID from '../../security/utils/verifyJWTTokenAndExtractMyID.js';
import isProduction from '../../shared/utils/isProduction.js';
import dataLoaders from '../dataLoaders/dataLoaders.js';
import permissions from '../permissions/permissions.js';
import schema from '../schemas/schema.js';

const graphQLMiddleware = createHandler({
  schema: applyMiddleware(schema, permissions),
  context: ({ raw }) => ({
    dataLoaders: dataLoaders(),
    myId: verifyJWTTokenAndExtractMyID((raw as Request).headers.authorization),
  }),
  validationRules: [...(isProduction() ? [NoSchemaIntrospectionCustomRule] : []), depthLimit(5)],
});

export default graphQLMiddleware;
