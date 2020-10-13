import { RequestHandler } from 'express';
import NEL from 'network-error-logging';

const networkErrorLoggingMiddleware = (): RequestHandler => {
  // Set header 'NEL'
  return NEL({
    report_to: 'default',
    max_age: 31536000,
    include_subdomains: true,
  });
};

export default networkErrorLoggingMiddleware;
