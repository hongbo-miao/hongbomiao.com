import { RequestHandler } from 'express';
import morgan from 'morgan';
import isProd from '../../app/utils/isProd';

// https://github.com/expressjs/morgan#predefined-formats
const FORMAT = isProd ? 'combined' : 'dev';

const morganMiddleware = (format: string = FORMAT): RequestHandler => {
  return morgan(format);
};

export default morganMiddleware;
