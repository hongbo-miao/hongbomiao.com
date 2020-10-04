import { RequestHandler } from 'express';
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import NEL from 'network-error-logging';

const networkErrorLoggingMiddleware = (): RequestHandler => {
  return NEL({
    report_to: 'default',
    max_age: 31536000,
    include_subdomains: true,
  });
};

export default networkErrorLoggingMiddleware;
