import express from 'express';
import request from 'supertest';
import rateLimitMiddleware from './rateLimit.middleware';

describe('rateLimitMiddleware', () => {
  const windowMs = 60 * 1000; // 60 sec
  const max = 2; // Requests per IP

  const app = express()
    .use(rateLimitMiddleware(windowMs, max))
    .get('/', (req, res) => {
      res.send('Hello, World!');
    });

  test('should limit rate', (done) => {
    request(app).get('/').expect(200).end(done);
    request(app).get('/').expect(200).end(done);
    request(app).get('/').expect(429).expect('Sorry, too many requests, please try again later.').end(done);
  });
});
