import { Request, Response } from 'express';

const reportCSPViolation = (req: Request, res: Response): void => {
  // eslint-disable-next-line no-console
  console.error('reportCSPViolation', req.body);
  res.status(200);
};

export default reportCSPViolation;
