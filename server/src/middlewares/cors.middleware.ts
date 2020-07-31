import cors, { CorsOptions } from 'cors';
import { NextFunction, Request, RequestHandler, Response } from 'express';

const corsMiddleware = (req: Request, res: Response, next: NextFunction): RequestHandler => {
  const nonProdWhiteList = ['http://localhost:3000', 'http://localhost:3001'];
  // const prodWhiteList = ['https://hongbomiao.com', 'https://www.hongbomiao.com'];
  // const whitelist = process.env.NODE_ENV === 'production' ? prodWhiteList : nonProdWhiteList;
  const whitelist = nonProdWhiteList;
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
