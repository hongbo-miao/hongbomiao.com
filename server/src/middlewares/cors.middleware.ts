import cors, { CorsOptions } from 'cors';
import { NextFunction, Request, RequestHandler, Response } from 'express';

import Config from '../config';

const corsMiddleware = (req: Request, res: Response, next: NextFunction): RequestHandler => {
  const whitelist = process.env.NODE_ENV === 'production' ? Config.prodWhitelist : Config.devWhitelist;
  const corsOptions: CorsOptions = {
    allowedHeaders: ['Authorization', 'Content-Type'],
    credentials: true,
    methods: ['GET', 'HEAD', 'PUT', 'POST', 'PATCH'],
    optionsSuccessStatus: 200,
    origin: (origin, callback) => {
      if (
        origin == null || // Do not block server-to-server requests and REST tools
        whitelist.includes(origin)
      ) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    },
  };
  return cors(corsOptions)(req, res, next);
};

export default corsMiddleware;
