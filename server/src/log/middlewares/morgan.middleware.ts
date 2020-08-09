import { RequestHandler } from 'express';
import morgan from 'morgan';
import isProduction from '../../shared/utils/isProduction';

// https://github.com/expressjs/morgan#predefined-formats
const FORMAT = isProduction ? 'combined' : 'dev';

const morganMiddleware = (format: string = FORMAT): RequestHandler => {
  return morgan(format);
};

export default morganMiddleware;
