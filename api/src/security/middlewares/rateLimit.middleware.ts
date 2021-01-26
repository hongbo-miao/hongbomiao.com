import { NextFunction, Request, RequestHandler, Response } from 'express';
import Redis from 'ioredis';
import { BurstyRateLimiter, RateLimiterMemory, RateLimiterRedis } from 'rate-limiter-flexible';

const BURST_POINTS_RATE = 2.5;
const BURST_DURATION_RATE = 10;

const POINTS = 300; // Number of points
const DURATION = 60; // Per 60 seconds
const INMEMORY_BLOCK_DURATION = 60; // If IP consume >= inmemoryBlockOnConsumed points, block it for 60 seconds in memory, so no requests go to Redis
const PROCESS_NUM = 1;

const rateLimitMiddleware = (
  redis: Redis.Redis,
  points: number = POINTS,
  duration: number = DURATION,
  burstPointsRate: number = BURST_POINTS_RATE,
  burstDurationRate: number = BURST_DURATION_RATE,
  inmemoryBlockDuration: number = INMEMORY_BLOCK_DURATION,
  processNum: number = PROCESS_NUM
): RequestHandler => {
  const memoryRateLimiter = new RateLimiterMemory({
    keyPrefix: 'memory',
    points: points / processNum,
    duration,
  });

  const redisRateLimiter = new RateLimiterRedis({
    keyPrefix: 'redis',
    storeClient: redis,
    points,
    duration,
    inmemoryBlockOnConsumed: points + 1,
    inmemoryBlockDuration,
    insuranceLimiter: memoryRateLimiter,
  });

  const burstMemoryRateLimiter = new RateLimiterMemory({
    keyPrefix: 'burstMemory',
    points: (points / processNum) * burstPointsRate,
    duration: duration * burstDurationRate,
  });

  const burstRedisRateLimiter = new RateLimiterRedis({
    keyPrefix: 'burstRedis',
    storeClient: redis,
    points: points * burstPointsRate,
    duration: duration * burstDurationRate,
    inmemoryBlockOnConsumed: points * burstPointsRate + 1,
    inmemoryBlockDuration,
    insuranceLimiter: burstMemoryRateLimiter,
  });

  const rateLimiter = new BurstyRateLimiter(redisRateLimiter, burstRedisRateLimiter);

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
