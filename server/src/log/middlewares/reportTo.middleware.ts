import { RequestHandler } from 'express';
import reportTo, { Endpoint } from 'report-to';
import config from '../../config';
import isProduction from '../../shared/utils/isProduction';

const REPORT_TO_ENDPOINT: Endpoint = isProduction()
  ? {
      url: config.reportURI.reportToURL,
      priority: 1,
    }
  : {
      url: `https://${config.host}:${config.port}/api/violation/report-to`,
      priority: 1,
    };

const reportToMiddleware = (reportToEndpoint: Endpoint = REPORT_TO_ENDPOINT): RequestHandler => {
  // Set header 'Report-To'
  return reportTo({
    groups: [
      {
        group: 'default',
        max_age: 31536000,
        include_subdomains: true,
        endpoints: [reportToEndpoint],
      },
    ],
  });
};

export default reportToMiddleware;
