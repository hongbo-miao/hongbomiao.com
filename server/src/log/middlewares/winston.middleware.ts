import { RequestHandler } from 'express';
import expressWinston from 'express-winston';
import winstonTransports from '../utils/winstonTransports';

const winstonMiddleware = (): RequestHandler => {
  return expressWinston.logger({
    transports: winstonTransports,
  });
};

export default winstonMiddleware;
