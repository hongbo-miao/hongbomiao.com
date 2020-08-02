import { promises as fsp } from 'fs';
import path from 'path';
import { NextFunction, Request, RequestHandler, Response } from 'express';
import helmet from 'helmet';
import getScriptSrcHashes from '../utils/getScriptSrcHashes';

const helmetMiddleware = async (req: Request, res: Response, next: NextFunction): Promise<RequestHandler> => {
  const indexPath = path.resolve(__dirname, '../../../client/build/index.html');
  const index = await fsp.readFile(indexPath, 'utf-8');
  const hashes = await getScriptSrcHashes(index);

  return helmet({
    // Content-Security-Policy
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
          "'sha256-KYbqXqOYv6xOvzbfRwXslg+GK2ebpVi0G6EzAvF6Yc8='",
          'https://edge.fullstory.com',
          'https://fullstory.com',
          'https://storage.googleapis.com',
          'https://www.google-analytics.com',
          'https://www.googletagmanager.com',
          ...hashes,
        ],
        styleSrc: ["'self'", 'https://fonts.googleapis.com'],
        // sandbox: ['allow-forms', 'allow-scripts'],
        // reportUri: '/report-violation',
        // objectSrc: ["'none'"],
        // upgradeInsecureRequests: true,
      },
    },

    // X-DNS-Prefetch-Control
    dnsPrefetchControl: {
      allow: false,
    },

    // X-Frame-Options
    frameguard: {
      action: 'deny',
    },

    // X-Powered-By
    hidePoweredBy: true,

    // Strict-Transport-Security
    hsts: {
      // Must be at least 1 year to be approved
      maxAge: 31536000,

      // Must be enabled to be approved
      includeSubDomains: true,
      preload: true,
    },

    // X-Download-Options
    ieNoOpen: true,

    // X-Content-Type-Options
    noSniff: true,

    // X-Permitted-Cross-Domain-Policies
    permittedCrossDomainPolicies: {
      permittedPolicies: 'none',
    },

    // Referrer-Policy
    referrerPolicy: {
      policy: 'no-referrer-when-downgrade',
    },

    // X-XSS-Protection
    xssFilter: true,
  })(req, res, next);
};

export default helmetMiddleware;
