import cors from 'cors';
import Config from '../config';
import isProd from '../utils/isProd';

const whitelist = isProd ? Config.prodWhitelist : Config.devWhitelist;

const corsMiddleware = cors({
  allowedHeaders: ['Authorization', 'Content-Type'],
  credentials: true,
  methods: ['GET', 'HEAD', 'PUT', 'POST', 'PATCH'],
  optionsSuccessStatus: 200,
  origin: (origin, callback) => {
    if (
      origin == null || // Server-to-server requests and REST tools
      origin === 'null' || // Safari reports CSP violation on localhost
      whitelist.includes(origin)
    ) {
      callback(null, true);
    } else {
      callback(new Error(`${origin} is not allowed by CORS.`));
    }
  },
});

export default corsMiddleware;
