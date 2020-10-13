import { RequestHandler } from 'express';
import reportTo from 'report-to';
import config from '../../config';
import isProduction from '../../shared/utils/isProduction';

const REPORT_TO_URL = isProduction()
  ? config.reportURI.reportToURL
  : `https://${config.domain}:${config.port}/api/violation/report-to`;

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
