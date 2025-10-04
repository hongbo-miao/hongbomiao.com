import { Request, Response } from 'express';
import logger from '../../log/utils/logger.js';

const reportTo = (req: Request, res: Response): void => {
  logger.warn(req.body, 'reportTo');
  res.sendStatus(200);
};

export default reportTo;
