import { RequestHandler } from 'express';
import redirectSSL from 'redirect-ssl';
import isProduction from '../../shared/utils/isProduction';

const redirectSSLMiddleware = (): RequestHandler => {
  return redirectSSL.create({
    enabled: isProduction(),
  });
};

export default redirectSSLMiddleware;
