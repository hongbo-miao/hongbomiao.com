import express from 'express';
// https://github.com/stipsan/ioredis-mock/pull/849
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import MockedRedis from 'ioredis-mock';
import request from 'supertest';
import rateLimitMiddleware from './rateLimit.middleware';

describe('rateLimitMiddleware', () => {
  const points = 2; // Number of points
  const duration = 60; // Per 60 seconds
  const burstPointsRate = 1.5;
  const burstDurationRate = 10;

  const redis = new MockedRedis();

  const app = express()
    .use(rateLimitMiddleware(redis, points, duration, burstPointsRate, burstDurationRate))
    .get('/', (req, res) => {
      res.send('Hello, World!');
    });

  test('should limit rate', async () => {
    // Consumed by rate limiter
    await request(app).get('/').expect(200);
    await request(app).get('/').expect(200);

    // Consumed by burst rate limiter
    await request(app).get('/').expect(200);
    await request(app).get('/').expect(200);
    await request(app).get('/').expect(200);

    await request(app).get('/').expect(429).expect('Sorry, too many requests, please try again later.');
  });
});
