import rateLimit, { Options } from 'express-rate-limit';

const rateLimitOptions: Options = {
  windowMs: 60 * 1000, // 60 sec
  max: 100, // requests per IP
  message: 'Sorry, too many requests, please try again later.',
};
const rateLimitMiddleware = rateLimit(rateLimitOptions);

export default rateLimitMiddleware;
