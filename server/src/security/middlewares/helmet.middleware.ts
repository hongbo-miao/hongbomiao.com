import { promises as fsp } from 'fs';
import path from 'path';
import { NextFunction, Request, Response } from 'express';
import helmet from 'helmet';
import getScriptSrcHashes from '../utils/getScriptSrcHashes';

const helmetMiddleware = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  const indexPath = path.resolve(__dirname, '../../dist/index.html');
  const index = await fsp.readFile(indexPath, 'utf-8');
  const hashes = await getScriptSrcHashes(index);

  return helmet({
    // Set 'Content-Security-Policy'
    contentSecurityPolicy: {
      directives: {
        baseUri: ["'none'"],
        defaultSrc: ["'none'"],
        connectSrc: [
          "'self'",
          'https://www.google-analytics.com',
          'https://stats.g.doubleclick.net',
          'https://rs.fullstory.com',
        ],
        formAction: ["'none'"],
        fontSrc: ["'self'", 'data:', 'https://fonts.gstatic.com'],
        frameAncestors: ["'none'"],
        imgSrc: [
          "'self'",
          'data:',
          'https://www.google-analytics.com',
          'https://stats.g.doubleclick.net',
          'https://www.google.com',
        ],
        manifestSrc: ["'self'"],
        scriptSrc: [
          "'self'",
          "'sha256-KYbqXqOYv6xOvzbfRwXslg+GK2ebpVi0G6EzAvF6Yc8='", // Google Tag Manager
          'https://edge.fullstory.com',
          'https://fullstory.com',
          'https://storage.googleapis.com',
          'https://www.google-analytics.com',
          'https://www.googletagmanager.com',
          ...hashes,
        ],
        styleSrc: ["'self'", 'https://fonts.googleapis.com'],
        // sandbox: ['allow-forms', 'allow-scripts'],
        reportUri: '/api/violation/report-csp-violation',
        // objectSrc: ["'none'"],
        // upgradeInsecureRequests: true,
      },
    },

    // Set 'X-DNS-Prefetch-Control: off'
    dnsPrefetchControl: {
      allow: false,
    },

    // Set 'Expect-CT: enforce, max-age=86400'
    expectCt: {
      maxAge: 86400,
      enforce: true,
    },

    // Set 'X-Frame-Options: DENY'
    frameguard: {
      action: 'deny',
    },

    // Remove 'X-Powered-By'
    hidePoweredBy: undefined,

    // Set 'Strict-Transport-Security: max-age=31536000; includeSubDomains; preload'
    hsts: {
      // Must be at least 1 year to be approved
      maxAge: 31536000,

      // Must be enabled to be approved
      includeSubDomains: true,
      preload: true,
    },

    // Set 'X-Download-Options: noopen'
    ieNoOpen: undefined,

    // Set 'X-Content-Type-Options: nosniff'
    noSniff: undefined,

    // Set 'X-Permitted-Cross-Domain-Policies: none'
    permittedCrossDomainPolicies: {
      permittedPolicies: 'none',
    },

    // Set 'Referrer-Policy: no-referrer'
    referrerPolicy: {
      policy: 'no-referrer',
    },

    // Set 'X-XSS-Protection: 0'
    xssFilter: undefined,
  })(req, res, next);
};

export default helmetMiddleware;
