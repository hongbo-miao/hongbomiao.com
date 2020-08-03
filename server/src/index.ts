import path from 'path';
import * as Sentry from '@sentry/node';
import bodyParser from 'body-parser';
import express from 'express';
import Config from './config';
import handleError from './controllers/error/handleError';
import corsMiddleware from './middlewares/cors.middleware';
import helmetMiddleware from './middlewares/helmet.middleware';
import morganMiddleware from './middlewares/morgan.middleware';
import rateLimitMiddleware from './middlewares/rateLimit.middleware';
import redirectSSLMiddleware from './middlewares/redirectSSL.middleware';
import apiRouter from './routers/api.router';

Sentry.init({
  dsn: Config.sentryDSN,
  environment: Config.nodeEnv,
});

const port = Config.port || 3001;
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
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../../client/build/index.html'));
});

app.use(Sentry.Handlers.errorHandler()); // The error handler must be before any other error middleware and after all controllers
app.use(handleError);

// eslint-disable-next-line no-console
app.listen(port, () => console.log(`Listening at port ${port}`));
