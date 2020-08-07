import http from 'http';
import app from './app';
import createHTTP2Server from './app/utils/createHTTP2Server';
import isProd from './app/utils/isProd';
import config from './config';
import createTerminus from './health/utils/createTerminus';
import initSentry from './log/utils/initSentry';
import logger from './log/utils/logger';

initSentry();

const server = isProd ? http.createServer(app) : createHTTP2Server(app);
server.listen(config.port, () => {
  logger.info(`NODE_ENV: ${config.nodeEnv}`);
  logger.info(`PORT: ${config.port}`);
});

createTerminus(server);
