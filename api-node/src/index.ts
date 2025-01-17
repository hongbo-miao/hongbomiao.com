// OpenTelemetry needs to be setup before importing other modules
import './reliability/utils/initTracer.js';
import http from 'http';
import { execute, subscribe } from 'graphql';
import { applyMiddleware } from 'graphql-middleware';
import { useServer } from 'graphql-ws/use/ws';
import { WebSocketServer } from 'ws';
import app from './app.js';
import config from './config.js';
import initPostgres from './dataSources/postgres/seeds/initPostgres.js';
import permissions from './graphQL/permissions/permissions.js';
import subscriptionSchema from './graphQL/schemas/subscriptionSchema.js';
import createTerminus from './health/utils/createTerminus.js';
import initSentry from './log/utils/initSentry.js';
import logger from './log/utils/logger.js';

initSentry();
initPostgres();

const { nodeEnv, port } = config;

const httpServer = http.createServer(app);
const wsServer = new WebSocketServer({
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
  wsServer,
);

createTerminus(httpServer);
