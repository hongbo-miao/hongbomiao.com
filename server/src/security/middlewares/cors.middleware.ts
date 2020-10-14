import cors from 'cors';
import { RequestHandler } from 'express';
import config from '../../config';
import logger from '../../log/utils/logger';
import meter from '../../log/utils/meter';
import isProduction from '../../shared/utils/isProduction';

const ALLOW_LIST = isProduction() ? config.prodCORSAllowList : config.devCORSAllowList;

const corsMiddleware = (allowList: ReadonlyArray<string> = ALLOW_LIST): RequestHandler => {
  const corsViolationCounter = meter.createCounter('corsViolationCounter', {
    description: 'Count CORS violations',
  });

  return cors({
    allowedHeaders: ['Authorization', 'Content-Type'],
    credentials: true,
    methods: ['GET', 'HEAD', 'PUT', 'POST', 'PATCH'],
    optionsSuccessStatus: 200,
    origin: (origin, callback) => {
      if (
        origin == null || // Server-to-server requests and REST tools
        allowList.includes(origin)
      ) {
        callback(null, true);
      } else {
        const labels = { origin };
        corsViolationCounter.bind(labels).add(1);
        const errMsg = `${origin} is not allowed by CORS.`;
        logger.warn(errMsg);
        callback(new Error(errMsg));
      }
    },
  });
};

export default corsMiddleware;
