import { Request, Response } from 'express';
import logger from '../../log/utils/logger';
import meter from '../../reliability/utils/meter';

const cspViolationCounter = meter.createCounter('cspViolationCounter', {
  description: 'Count CSP violations',
});

const reportCSPViolation = (req: Request, res: Response): void => {
  logger.warn(req.body, 'reportCSPViolation');
  const labels = req.body['csp-report'];
  cspViolationCounter.add(1, labels);
  res.sendStatus(200);
};

export default reportCSPViolation;
