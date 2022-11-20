import { RequestHandler } from 'express';
import { expressjwt as jwt } from 'express-jwt';
import config from '../../config';

const authMiddleware = (): RequestHandler => {
  return jwt({
    secret: config.jwtSecret,
    algorithms: ['HS256'],
  });
};

export default authMiddleware;
