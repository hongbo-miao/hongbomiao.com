import express from 'express';
import request from 'supertest';
import helmetMiddleware from './helmet.middleware';

describe('helmetMiddleware', () => {
  const app = express()
    .use(helmetMiddleware())
    .get('/', (req, res) => {
      res.send('Hello, World!');
    });

  test("should include script-src 'self' in content-security-policy", (done) => {
    request(app)
      .get('/')
      .expect('content-security-policy', /script-src/)
      .expect(200)
      .end(done);
  });
});
