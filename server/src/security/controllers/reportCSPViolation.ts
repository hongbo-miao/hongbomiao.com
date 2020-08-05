import { Request, Response } from 'express';
import logger from '../../log/utils/logger';

const reportCSPViolation = (req: Request, res: Response): void => {
  logger.error('reportCSPViolation', req.body);
  res.status(200);
};

export default reportCSPViolation;
