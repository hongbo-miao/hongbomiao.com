import cors from 'cors';
import { RequestHandler } from 'express';
import isProd from '../../app/utils/isProd';
import config from '../../config';

const ALLOW_LIST = isProd ? config.prodCORSAllowList : config.devCORSAllowList;

const corsMiddleware = (allowList: string[] = ALLOW_LIST): RequestHandler => {
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
        callback(new Error(`${origin} is not allowed by CORS.`));
      }
    },
  });
};

export default corsMiddleware;
