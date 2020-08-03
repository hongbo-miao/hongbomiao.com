import { NextFunction, Request, Response } from 'express';

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const handleError = async (err: Error, req: Request, res: Response, next: NextFunction): Promise<void> => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  const { sentry } = res;
  res.statusCode = 500;
  res.end(sentry);
};

export default handleError;
