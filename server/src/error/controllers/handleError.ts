import { NextFunction, Request, Response } from 'express';

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const handleError = (err: Error, req: Request, res: Response, next: NextFunction): void => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  switch (err.code) {
    case 'EBADCSRFTOKEN': {
      res.statusCode = 403;
      break;
    }
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
