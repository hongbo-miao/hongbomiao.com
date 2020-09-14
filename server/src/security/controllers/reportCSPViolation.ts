import { Request, Response } from 'express';
import logger from '../../log/utils/logger';
import meter from '../../log/utils/meter';

const cspViolationCounter = meter.createCounter('cspViolationCounter', {
  description: 'Count CSP violations',
});

const reportCSPViolation = (req: Request, res: Response): void => {
  logger.warn('reportCSPViolation', req.body);
  const labels = req.body['csp-report'];
  cspViolationCounter.bind(labels).add(1);
  res.sendStatus(200);
};

export default reportCSPViolation;
