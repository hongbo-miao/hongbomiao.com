import express from 'express';
import request from 'supertest';
import corsMiddleware from './cors.middleware';

describe('corsMiddleware', () => {
  const allowList = ['https://hongbomiao.com'];

  const app = express()
    .use(corsMiddleware(allowList))
    .get('/', (req, res) => {
      res.send('Hello, World!');
    });

  test('should succeed if origin is undefined for server-to-server requests and REST tools)', (done) => {
    request(app).get('/').expect(200).end(done);
  });

  test('should succeed if origin is in whitelist', (done) => {
    const allowOrigin = allowList[0];
    request(app).get('/').set('Origin', allowOrigin).expect(200).end(done);
  });

  test('should fail if origin is not in whitelist', (done) => {
    const nonAllowOrigin = 'https://evil.com';
    request(app).get('/').set('Origin', nonAllowOrigin).expect(500).end(done);
  });
});
