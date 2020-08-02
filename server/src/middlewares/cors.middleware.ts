import cors from 'cors';

import Config from '../config';

const whitelist = process.env.NODE_ENV === 'production' ? Config.prodWhitelist : Config.devWhitelist;

const corsMiddleware = cors({
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
});

export default corsMiddleware;
