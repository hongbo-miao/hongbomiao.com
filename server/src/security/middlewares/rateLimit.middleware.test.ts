import express from 'express';
// https://github.com/stipsan/ioredis-mock/pull/849
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import MockedRedis from 'ioredis-mock';
import request from 'supertest';
import rateLimitMiddleware from './rateLimit.middleware';

describe('rateLimitMiddleware', () => {
  const duration = 60; // Seconds
  const points = 2; // Requests per IP

  const redis = new MockedRedis({
    enableOfflineQueue: false,
  });

  const app = express()
    .use(rateLimitMiddleware(redis, duration, points))
    .get('/', (req, res) => {
      res.send('Hello, World!');
    });

  test('should limit rate', async () => {
    await request(app).get('/').expect(200);
    await request(app).get('/').expect(200);
    await request(app).get('/').expect(429).expect('Sorry, too many requests, please try again later.');
  });
});
