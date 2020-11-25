// OpenTelemetry need to be setup before importing other modules
import './shared/utils/initTracer';
import http from 'http';
import { execute, subscribe } from 'graphql';
import { applyMiddleware } from 'graphql-middleware';
import { useServer } from 'graphql-ws/lib/use/ws';
import WebSocket from 'ws';
import app from './app';
import config from './config';
import initPostgres from './database/postgres/seeds/initPostgres';
import permissions from './graphQL/permissions/permissions';
import subscriptionSchema from './graphQL/schemas/subscriptionSchema';
import createTerminus from './health/utils/createTerminus';
import initSentry from './log/utils/initSentry';
import logger from './log/utils/logger';
import createHTTP2Server from './shared/utils/createHTTP2Server';
import isProduction from './shared/utils/isProduction';

initSentry();
initPostgres();

const { nodeEnv, port } = config;

const httpServer = isProduction() ? http.createServer(app) : createHTTP2Server(app);
const wsServer = new WebSocket.Server({
  server: httpServer,
  path: '/graphql',
});

httpServer.listen(port, () => {
  logger.info({ nodeEnv, port }, 'env');
});

useServer(
  {
    schema: applyMiddleware(subscriptionSchema, permissions),
    execute,
    subscribe,
  },
  wsServer
);

createTerminus(httpServer);
