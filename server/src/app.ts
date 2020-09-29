import path from 'path';
import * as Sentry from '@sentry/node';
import bodyParser from 'body-parser';
import express from 'express';
import Redis from 'ioredis';
import favicon from 'serve-favicon';
import Config from './Config';
import handleError from './error/controllers/handleError';
import graphQLMiddleware from './graphQL/middlewares/graphQL.middleware';
import incomingRequestCounterMiddleware from './log/middlewares/incomingRequestCounter.middleware';
import pinoMiddleware from './log/middlewares/pino.middleware';
import sendIndexPage from './page/controllers/sendIndexPage';
import corsMiddleware from './security/middlewares/cors.middleware';
import helmetMiddleware from './security/middlewares/helmet.middleware';
import rateLimitMiddleware from './security/middlewares/rateLimit.middleware';
import apiRouter from './shared/routers/api.router';

const redis = new Redis(Config.redisOptions);

const app = express()
  .use(Sentry.Handlers.requestHandler()) // Must be the first middleware on the app
  .use(bodyParser.json())
  .use(pinoMiddleware())
  .use(incomingRequestCounterMiddleware())
  .use(corsMiddleware())
  .use(helmetMiddleware())
  .get('/', sendIndexPage)
  .use(favicon(path.join(__dirname, '../dist/favicon.ico')))
  .use(express.static(path.join(__dirname, '../dist')))
  .use(rateLimitMiddleware(redis))
  .use('/graphql', graphQLMiddleware)
  .use('/api', apiRouter)
  .use(Sentry.Handlers.errorHandler()) // Must be before any other error middleware and after all controllers
  .use(handleError);

export default app;
