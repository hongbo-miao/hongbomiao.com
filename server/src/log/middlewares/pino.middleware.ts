import { RequestHandler } from 'express';
import pinoHTTP from 'pino-http';
import Config from '../../Config';
import logger from '../utils/logger';

const pinoMiddleware = (): RequestHandler => {
  return pinoHTTP({
    autoLogging: !Config.shouldHideHTTPLog,
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
