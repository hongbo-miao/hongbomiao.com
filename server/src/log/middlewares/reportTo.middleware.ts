import { RequestHandler } from 'express';
import reportTo from 'report-to';
import config from '../../config';

const REPORT_TO_URL = config.reportURI.reportToURL;

const reportToMiddleware = (reportToURL: string = REPORT_TO_URL): RequestHandler => {
  // Set header 'Report-To'
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
