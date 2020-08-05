import path from 'path';
import * as Sentry from '@sentry/node';
import bodyParser from 'body-parser';
import express from 'express';
import graphQLMiddleware from './app/middlewares/graphQL.middleware';
import apiRouter from './app/routers/api.router';
import handleError from './error/controllers/handleError';
import morganMiddleware from './log/middlewares/morgan.middleware';
import sendIndexPage from './page/controllers/sendIndexPage';
import corsMiddleware from './security/middlewares/cors.middleware';
import helmetMiddleware from './security/middlewares/helmet.middleware';
import rateLimitMiddleware from './security/middlewares/rateLimit.middleware';
import redirectSSLMiddleware from './security/middlewares/redirectSSL.middleware';

const app = express()
  .use(Sentry.Handlers.requestHandler()) // The request handler must be the first middleware on the app
  .use(morganMiddleware)
  .use(corsMiddleware)
  .use(helmetMiddleware)
  .use(redirectSSLMiddleware)
  .use(express.static(path.join(__dirname, '../dist')))
  .get('/', sendIndexPage)
  .use(rateLimitMiddleware)
  .use(bodyParser.json())
  .use('/graphql', graphQLMiddleware)
  .use('/api', apiRouter)
  .use(Sentry.Handlers.errorHandler()) // The error handler must be before any other error middleware and after all controllers
  .use(handleError);

export default app;
