import rateLimit from 'express-rate-limit';

const rateLimitMiddleware = rateLimit({
  windowMs: 60 * 1000, // 60 sec
  max: 100, // requests per IP
  message: 'Sorry, too many requests, please try again later.',
});

export default rateLimitMiddleware;
