import { RequestHandler } from 'express';
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import reportTo from 'report-to';
import config from '../../config';

const REPORT_TO_URL = config.reportURI.reportToURL;

const reportToMiddleware = (reportToURL: string = REPORT_TO_URL): RequestHandler => {
  return reportTo({
    groups: [
      {
        group: 'default',
        max_age: 31536000,
        include_subdomains: true,
        endpoints: [
          {
            url: reportToURL,
            priority: 1,
          },
        ],
      },
    ],
  });
};

export default reportToMiddleware;
