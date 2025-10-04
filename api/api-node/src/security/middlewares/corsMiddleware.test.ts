import express from 'express';
import request from 'supertest';
import corsMiddleware from './corsMiddleware.js';

describe('corsMiddleware', () => {
  const allowOrigins = ['https://www.hongbomiao.com'];

  const app = express()
    .use(corsMiddleware(allowOrigins))
    .get('/', (req, res) => {
      res.send('Hello, World!');
    });

  test('should succeed if origin is undefined for server-to-server requests and REST tools)', async () => {
    await request(app).get('/').expect(200);
  });

  test('should succeed if origin is in whitelist', async () => {
    const allowOrigin = allowOrigins[0];
    await request(app).get('/').set('Origin', allowOrigin).expect(200);
  });

  test('should fail if origin is not in whitelist', async () => {
    const nonAllowOrigin = 'https://evil.com';
    await request(app).get('/').set('Origin', nonAllowOrigin).expect(500);
  });
});
