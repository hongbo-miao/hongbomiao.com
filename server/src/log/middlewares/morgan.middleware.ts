import morgan from 'morgan';
import isProd from '../../app/utils/isProd';

// https://github.com/expressjs/morgan#predefined-formats
const format = isProd ? 'combined' : 'dev';
const morganMiddleware = morgan(format);

export default morganMiddleware;
