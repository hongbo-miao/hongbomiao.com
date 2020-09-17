import { RequestHandler } from 'express';
import expressWinston from 'express-winston';
import winston from 'winston';
import winstonTransports from '../utils/winstonTransports';

const winstonMiddleware = (): RequestHandler => {
  return expressWinston.logger({
    transports: winstonTransports,
    format: winston.format.combine(winston.format.colorize(), winston.format.json()),
  });
};

export default winstonMiddleware;
