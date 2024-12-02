import cors from 'cors';
import { RequestHandler } from 'express';
import config from '../../config.js';
import logger from '../../log/utils/logger.js';
import meter from '../../reliability/utils/meter.js';
import isProduction from '../../shared/utils/isProduction.js';

const ALLOW_LIST = isProduction() ? config.prodCORSAllowOrigins : config.devCORSAllowOrigins;

const corsMiddleware = (allowOrigins: ReadonlyArray<string> = ALLOW_LIST): RequestHandler => {
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
        allowOrigins.includes(origin)
      ) {
        callback(null, true);
      } else {
        const labels = { origin };
        corsViolationCounter.add(1, labels);
        const errMsg = `${origin} is not allowed by CORS.`;
        logger.warn({ errMsg }, 'cors');
        callback(new Error(errMsg));
      }
    },
  });
};

export default corsMiddleware;
