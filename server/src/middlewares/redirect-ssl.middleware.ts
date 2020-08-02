import redirectSSL from 'redirect-ssl';

const redirectSSLMiddleware = redirectSSL.create({
  enabled: process.env.NODE_ENV === 'production',
});

export default redirectSSLMiddleware;
