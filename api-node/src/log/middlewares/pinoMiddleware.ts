import { RequestHandler } from 'express';
import pinoHTTP from 'pino-http';
import config from '../../config.js';
import logger from '../utils/logger.js';

const pinoMiddleware = (): RequestHandler => {
  return pinoHTTP({
    autoLogging: !config.shouldHideHttpLog,
    logger,
    serializers: {
      req: (req) => {
        req.body = req.raw.body;
        return req;
      },
    },
  });
};

export default pinoMiddleware;
