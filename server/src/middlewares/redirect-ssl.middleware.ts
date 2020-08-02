import redirectSSL from 'redirect-ssl';

import isProd from '../utils/isProd';

const redirectSSLMiddleware = redirectSSL.create({
  enabled: isProd,
});

export default redirectSSLMiddleware;
