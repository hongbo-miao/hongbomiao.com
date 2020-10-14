import { RequestHandler } from 'express';
import reportTo from 'report-to';
import config from '../../config';
import isProduction from '../../shared/utils/isProduction';

const reportToMiddleware = (): RequestHandler => {
  // Set header 'Report-To'
  return reportTo({
    groups: [
      {
        group: 'default',
        max_age: 31536000,
        include_subdomains: true,
        endpoints: [
          ...(isProduction()
            ? [
                {
                  url: config.reportURI.reportToURL,
                  priority: 1,
                },
              ]
            : []),
          {
            url: `https://${config.domain}:${config.port}/api/violation/report-to`,
            priority: 1,
          },
        ],
      },
    ],
  });
};

export default reportToMiddleware;
