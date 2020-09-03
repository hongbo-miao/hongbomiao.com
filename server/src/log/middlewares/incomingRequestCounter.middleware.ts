import { RequestHandler } from 'express';
import meter from '../utils/meter';

const incomingRequestCounterMiddleware = (): RequestHandler => {
  const incomingRequestCounter = meter.createCounter('incomingRequestCounter', {
    description: 'Count incoming requests',
  });

  return (req, res, next) => {
    const labels = { path: req.path };
    incomingRequestCounter.bind(labels).add(1);
    next();
  };
};

export default incomingRequestCounterMiddleware;
