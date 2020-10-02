import './shared/utils/initTracer';
import http from 'http';
import app from './app';
import config from './config';
import initPostgres from './database/postgres/seeds/initPostgres';
import createTerminus from './health/utils/createTerminus';
import initSentry from './log/utils/initSentry';
import logger from './log/utils/logger';
import createHTTP2Server from './shared/utils/createHTTP2Server';
import isProduction from './shared/utils/isProduction';

initSentry();
initPostgres();

const server = isProduction() ? http.createServer(app) : createHTTP2Server(app);
const { nodeEnv, port } = config;

server.listen(port, () => {
  logger.info(
    {
      nodeEnv,
      port,
    },
    'env'
  );
});

createTerminus(server);
