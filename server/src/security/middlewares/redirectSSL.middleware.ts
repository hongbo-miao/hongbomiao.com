import { RequestHandler } from 'express';
import redirectSSL from 'redirect-ssl';
import isProd from '../../app/utils/isProd';

const redirectSSLMiddleware = (): RequestHandler => {
  return redirectSSL.create({
    enabled: isProd,
  });
};

export default redirectSSLMiddleware;
