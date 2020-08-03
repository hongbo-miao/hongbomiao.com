import { Request, Response } from 'express';

const reportCSPViolation = async (req: Request, res: Response): Promise<void> => {
  console.log('req.body', req.body);
  res.status(200);
};

export default reportCSPViolation;
