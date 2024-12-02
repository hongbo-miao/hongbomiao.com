import * as Sentry from '@sentry/node';
import cookieParser from 'cookie-parser';
import express from 'express';
import responseTime from 'response-time';
import favicon from 'serve-favicon';
import handleError from './error/controllers/handleError.js';
import graphQLMiddleware from './graphQL/middlewares/graphQLMiddleware.js';
import graphQLUploadMiddleware from './graphQL/middlewares/graphQLUploadMiddleware.js';
import incomingRequestCounterMiddleware from './log/middlewares/incomingRequestCounterMiddleware.js';
import networkErrorLoggingMiddleware from './log/middlewares/networkErrorLoggingMiddleware.js';
import pinoMiddleware from './log/middlewares/pinoMiddleware.js';
import reportToMiddleware from './log/middlewares/reportToMiddleware.js';
import indexRouter from './page/routers/indexRouter.js';
import redirectToIndexRouter from './page/routers/redirectToIndexRouter.js';
import corsMiddleware from './security/middlewares/corsMiddleware.js';
import helmetMiddleware from './security/middlewares/helmetMiddleware.js';
import rateLimitMiddleware from './security/middlewares/rateLimitMiddleware.js';
import redis from './security/utils/redis.js';
import apiRouter from './shared/routers/apiRouter.js';

const app = express()
  .use(express.json())
  .use(cookieParser())
  .use(pinoMiddleware())
  .use(incomingRequestCounterMiddleware())
  .use(corsMiddleware())
  .use(reportToMiddleware())
  .use(networkErrorLoggingMiddleware())
  .use(helmetMiddleware())
  .use(responseTime())
  .use(indexRouter)
  .use(favicon('public/favicon.ico'))
  .use(express.static('public', { maxAge: '1y' }))
  .use(rateLimitMiddleware(redis))
  .use('/graphql', graphQLUploadMiddleware, graphQLMiddleware)
  .use('/api', apiRouter)
  .use(redirectToIndexRouter);
Sentry.setupExpressErrorHandler(app) // Must be after all routes and before any and other error-handling middlewares
app.use(handleError)

export default app;
