import path from 'path';
import * as Sentry from '@sentry/node';
import bodyParser from 'body-parser';
import express from 'express';
import Config from './config';
import handleError from './controllers/error/handleError';
import sendIndexPage from './controllers/page/sendIndexPage';
import corsMiddleware from './middlewares/cors.middleware';
import helmetMiddleware from './middlewares/helmet.middleware';
import morganMiddleware from './middlewares/morgan.middleware';
import rateLimitMiddleware from './middlewares/rateLimit.middleware';
import redirectSSLMiddleware from './middlewares/redirectSSL.middleware';
import apiRouter from './routers/api.router';
import initSentry from './utils/initSentry';
import logger from './utils/logger';

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

app.use(express.static(path.join(__dirname, '../../client/build')));
app.get('/', sendIndexPage);

app.use(Sentry.Handlers.errorHandler()); // The error handler must be before any other error middleware and after all controllers
app.use(handleError);

app.listen(Config.port, () => logger.info(`Listening at port ${Config.port}`));
