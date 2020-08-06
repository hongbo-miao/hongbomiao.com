import cors from 'cors';
import isProd from '../../app/utils/isProd';
import config from '../../config';

const corsAllowList = isProd ? config.prodCORSAllowList : config.devCORSAllowList;

const corsMiddleware = cors({
  allowedHeaders: ['Authorization', 'Content-Type'],
  credentials: true,
  methods: ['GET', 'HEAD', 'PUT', 'POST', 'PATCH'],
  optionsSuccessStatus: 200,
  origin: (origin, callback) => {
    if (
      origin == null || // Server-to-server requests and REST tools
      corsAllowList.includes(origin)
    ) {
      callback(null, true);
    } else {
      callback(new Error(`${origin} is not allowed by CORS.`));
    }
  },
});

export default corsMiddleware;
