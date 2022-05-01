import { RequestHandler } from 'express';
import meter from '../../reliability/utils/meter';

const incomingRequestCounterMiddleware = (): RequestHandler => {
  const incomingRequestCounter = meter.createCounter('incomingRequestCounter', {
    description: 'Count incoming requests',
  });

  return (req, res, next) => {
    const labels = { path: req.path };
    incomingRequestCounter.add(1, labels);
    next();
  };
};

export default incomingRequestCounterMiddleware;
