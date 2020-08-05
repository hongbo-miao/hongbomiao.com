import redirectSSL from 'redirect-ssl';
import isProd from '../../app/utils/isProd';

const redirectSSLMiddleware = redirectSSL.create({
  enabled: isProd,
});

export default redirectSSLMiddleware;
