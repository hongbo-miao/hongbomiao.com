import { RequestHandler } from 'express';
import rateLimit from 'express-rate-limit';

const WINDOW_MS = 60 * 1000; // 60 sec
const MAX = 100; // Requests per IP

const rateLimitMiddleware = (windowMs: number = WINDOW_MS, max: number = MAX): RequestHandler => {
  return rateLimit({
    windowMs,
    max,
    message: 'Sorry, too many requests, please try again later.',
  });
};

export default rateLimitMiddleware;
