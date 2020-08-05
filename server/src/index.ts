import path from 'path';
import * as Sentry from '@sentry/node';
import bodyParser from 'body-parser';
import express from 'express';
import apiRouter from './app/routers/api.router';
import createHTTP2Server from './app/utils/createHTTP2Server';
import isProd from './app/utils/isProd';
import Config from './config';
import handleError from './error/controllers/handleError';
import morganMiddleware from './log/middlewares/morgan.middleware';
import initSentry from './log/utils/initSentry';
import printStatus from './log/utils/printStatus';
import sendIndexPage from './page/controllers/sendIndexPage';
import corsMiddleware from './security/middlewares/cors.middleware';
import helmetMiddleware from './security/middlewares/helmet.middleware';
import rateLimitMiddleware from './security/middlewares/rateLimit.middleware';
import redirectSSLMiddleware from './security/middlewares/redirectSSL.middleware';

initSentry();

const app = express();

app.use(Sentry.Handlers.requestHandler()); // The request handler must be the first middleware on the app
app.use(morganMiddleware);
app.use(corsMiddleware);
app.use(rateLimitMiddleware);
app.use(helmetMiddleware);
app.use(redirectSSLMiddleware);
app.use(bodyParser.json());

app.use('/api', apiRouter);

app.use(express.static(path.join(__dirname, '../dist')));
app.get('/', sendIndexPage);

app.use(Sentry.Handlers.errorHandler()); // The error handler must be before any other error middleware and after all controllers
app.use(handleError);

if (isProd) {
  app.listen(Config.port, printStatus);
} else {
  const http2Server = createHTTP2Server(app);
  http2Server.listen(Config.port, printStatus);
}
