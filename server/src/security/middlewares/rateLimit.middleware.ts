import { NextFunction, Request, RequestHandler, Response } from 'express';
import Redis from 'ioredis';
import { RateLimiterMemory, RateLimiterRedis } from 'rate-limiter-flexible';
import Config from '../../Config';

const DURATION = 60; // Number of points
const POINTS = 300; // Per 60 seconds
const INMEMORY_BLOCK_ON_CONSUMED = 301; // If userId or IP consume >=301 points per minute
const INMEMORY_BLOCK_DURATION = 60; // Block it for a minute in memory, so no requests go to Redis
const PROCESS_NUM = 1;
const REDIS = new Redis(Config.redisOptions);

const rateLimitMiddleware = (
  redis: Redis.Redis = REDIS,
  duration: number = DURATION,
  points: number = POINTS,
  inmemoryBlockOnConsumed: number = INMEMORY_BLOCK_ON_CONSUMED,
  inmemoryBlockDuration: number = INMEMORY_BLOCK_DURATION,
  processNum: number = PROCESS_NUM
): RequestHandler => {
  const memoryRateLimiter = new RateLimiterMemory({
    points: points / processNum,
    duration,
  });

  const redisRateLimiter = new RateLimiterRedis({
    storeClient: redis,
    keyPrefix: 'middleware',
    duration,
    points,
    inmemoryBlockOnConsumed,
    inmemoryBlockDuration,
    insuranceLimiter: memoryRateLimiter,
  });

  return (req: Request, res: Response, next: NextFunction) => {
    return redisRateLimiter
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
