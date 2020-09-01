import { NextFunction, Request, RequestHandler, Response } from 'express';
import Redis from 'ioredis';
import { RateLimiterRedis } from 'rate-limiter-flexible';
import Config from '../../Config';

const DURATION = 60; // Seconds
const POINTS = 5; // Requests per IP
const REDIS = new Redis(Config.redisOptions);

const rateLimitMiddleware = (
  redis: Redis.Redis = REDIS,
  duration: number = DURATION,
  points: number = POINTS
): RequestHandler => {
  const rateLimiter = new RateLimiterRedis({
    storeClient: redis,
    keyPrefix: 'middleware',
    duration,
    points,
  });

  return (req: Request, res: Response, next: NextFunction) => {
    return rateLimiter
      .consume(req.ip)
      .then(() => {
        next();
      })
      .catch(() => {
        res.status(429).send('Sorry, too many requests, please try again later.');
      });
  };
};

export default rateLimitMiddleware;
