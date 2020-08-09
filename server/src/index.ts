import http from 'http';
import Config from './Config';
import app from './app';
import createHTTP2Server from './app/utils/createHTTP2Server';
import isProduction from './app/utils/isProduction';
import createTerminus from './health/utils/createTerminus';
import initSentry from './log/utils/initSentry';
import logger from './log/utils/logger';

initSentry();

const server = isProduction ? http.createServer(app) : createHTTP2Server(app);
server.listen(Config.port, () => {
  logger.info(`NODE_ENV: ${Config.nodeEnv}`);
  logger.info(`PORT: ${Config.port}`);
});

createTerminus(server);
