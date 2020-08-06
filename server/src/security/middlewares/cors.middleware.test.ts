import express, { Request, Response } from 'express';
import request from 'supertest';
import config from '../../config';
import corsMiddleware from './cors.middleware';

describe('corsMiddleware', () => {
  const nonAllowListOrigin = 'https://evil.com';
  const allowListOrigin = config.devCORSAllowList[0];

  const app = express()
    .use(corsMiddleware)
    .get('/', (req: Request, res: Response) => {
      res.send('Hello, World!');
    });

  test('should succeed if origin is undefined for server-to-server requests and REST tools)', (done: jest.DoneCallback) => {
    request(app).get('/').expect(200).end(done);
  });

  test('should succeed if origin is in whitelist', (done: jest.DoneCallback) => {
    request(app).get('/').set('Origin', allowListOrigin).expect(200).end(done);
  });

  test('should fail if origin is not in whitelist', (done: jest.DoneCallback) => {
    request(app).get('/').set('Origin', nonAllowListOrigin).expect(500).end(done);
  });
});
