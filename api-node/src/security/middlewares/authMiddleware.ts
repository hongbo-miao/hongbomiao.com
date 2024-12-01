import { RequestHandler } from 'express';
import { expressjwt as jwt } from 'express-jwt';
import config from '../../config';

const authMiddleware = (): RequestHandler => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  return jwt({
    secret: config.jwtSecret,
    algorithms: ['HS256'],
  });
};

export default authMiddleware;
