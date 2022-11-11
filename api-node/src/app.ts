import path from 'path';
import * as Sentry from '@sentry/node';
import cookieParser from 'cookie-parser';
import express from 'express';
import requestID from 'express-request-id';
import responseTime from 'response-time';
import favicon from 'serve-favicon';
import handleError from './error/controllers/handleError';
import graphQLMiddleware from './graphQL/middlewares/graphQLMiddleware';
import graphQLUploadMiddleware from './graphQL/middlewares/graphQLUploadMiddleware';
import incomingRequestCounterMiddleware from './log/middlewares/incomingRequestCounterMiddleware';
import networkErrorLoggingMiddleware from './log/middlewares/networkErrorLoggingMiddleware';
import pinoMiddleware from './log/middlewares/pinoMiddleware';
import reportToMiddleware from './log/middlewares/reportToMiddleware';
import indexRouter from './page/routers/indexRouter';
import redirectToIndexRouter from './page/routers/redirectToIndexRouter';
import corsMiddleware from './security/middlewares/corsMiddleware';
import helmetMiddleware from './security/middlewares/helmetMiddleware';
import rateLimitMiddleware from './security/middlewares/rateLimitMiddleware';
import redis from './security/utils/redis';
import apiRouter from './shared/routers/apiRouter';

const app = express()
  .use(Sentry.Handlers.requestHandler()) // Must be the first middleware on the app
  .use(express.json())
  .use(cookieParser())
  .use(pinoMiddleware())
  .use(incomingRequestCounterMiddleware())
  .use(corsMiddleware())
  .use(reportToMiddleware())
  .use(networkErrorLoggingMiddleware())
  .use(helmetMiddleware())
  .use(requestID())
  .use(responseTime())
  .use(indexRouter)
  .use(favicon(path.join(__dirname, '../public/favicon.ico')))
  .use(express.static(path.join(__dirname, '../public'), { maxAge: '1y' }))
  .use(rateLimitMiddleware(redis))
  .use('/graphql', graphQLUploadMiddleware, graphQLMiddleware)
  .use('/api', apiRouter)
  .use(redirectToIndexRouter)
  .use(Sentry.Handlers.errorHandler()) // Must be before any other error middleware and after all controllers
  .use(handleError);

export default app;
