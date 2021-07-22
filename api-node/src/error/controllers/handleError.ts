import { NextFunction, Request, Response } from 'express';
import logger from '../../log/utils/logger';

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const handleError = (err: Error, req: Request, res: Response, next: NextFunction): void => {
  logger.error({ err }, 'handleError');

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  switch (err.code) {
    // multer
    case 'LIMIT_FILE_SIZE': {
      res.statusCode = 400;
      break;
    }

    // express-jwt
    case 'invalid_token': {
      res.statusCode = 401;
      break;
    }

    // express-jwt, csurf
    case 'credentials_required':
    case 'EBADCSRFTOKEN': {
      res.statusCode = 403;
      break;
    }

    // timeout
    case 'ETIMEDOUT': {
      res.statusCode = 408;
      break;
    }

    default: {
      res.statusCode = 500;
    }
  }

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  res.end(res.sentry);
};

export default handleError;
