import { NextFunction, Request, RequestHandler, Response } from 'express';
import helmet from 'helmet';
import lodashUniq from 'lodash.uniq';
import config from '../../config';
import isProduction from '../../shared/utils/isProduction';
import createCSPNonce from '../utils/createCSPNonce';

const CSP_CONNECT_SRC = isProduction() ? config.prodCSPConnectSrc : config.devCSPConnectSrc;
const CSP_REPORT_URI = isProduction() ? config.reportURI.cspReportURI : '/api/violation/report-csp-violation';

const helmetMiddleware = (
  cspConnectSrc: ReadonlyArray<string> = CSP_CONNECT_SRC,
  cspReportURI: string = CSP_REPORT_URI,
): RequestHandler => {
  return (req: Request, res: Response, next: NextFunction) => {
    const cspNonce = createCSPNonce();
    res.locals.cspNonce = cspNonce;

    return helmet({
      // Set header 'Content-Security-Policy'
      contentSecurityPolicy: {
        directives: {
          /* Fetch directives */
          connectSrc: lodashUniq([
            "'self'",

            // FullStory
            'https://rs.fullstory.com',

            // Sentry
            'https://o379185.ingest.sentry.io',

            // Google Analytics 4
            'https://analytics.google.com',

            // Universal Analytics (Google Analytics)
            'https://stats.g.doubleclick.net',
            'https://www.google-analytics.com',

            ...cspConnectSrc,
          ]),
          defaultSrc: ["'none'"],
          fontSrc: lodashUniq([
            "'self'",
            'data:',

            // Google Tag Manager's Preview Mode
            'https://fonts.gstatic.com',
          ]),
          frameSrc: lodashUniq([
            // Google Ads remarketing
            'https://bid.g.doubleclick.net',
          ]),
          imgSrc: lodashUniq([
            "'self'",
            'data:',

            // Favicon on bare domain
            'https://hongbomiao.com',

            // FullStory
            'https://rs.fullstory.com',

            // Google Ads conversions
            'https://googleads.g.doubleclick.net',
            'https://www.google.com',

            // Google Ads remarketing
            'https://www.google.com',

            // Google Tag Manager's Preview Mode
            'https://ssl.gstatic.com',
            'https://www.googletagmanager.com',

            // Universal Analytics (Google Analytics)
            'https://stats.g.doubleclick.net',
            'https://www.google-analytics.com',
          ]),
          manifestSrc: ["'self'"],
          mediaSrc: ["'self'"],
          objectSrc: ["'none'"],
          scriptSrc: lodashUniq([
            /* Content Security Policy Level 3 */
            "'strict-dynamic'",
            `'nonce-${cspNonce}'`,

            /* Content Security Policy Level 2 (backward compatible) */
            "'self'",

            // Cloudflare Browser Insights
            'https://static.cloudflareinsights.com',

            // FullStory
            'https://edge.fullstory.com',
            'https://www.fullstory.com',
            'https://fullstory.com',

            // Workbox
            'https://storage.googleapis.com',

            // Google Ads conversions
            'https://www.google.com',
            'https://www.googleadservices.com',

            // Google Ads remarketing
            'https://googleads.g.doubleclick.net',
            'https://www.google.com',
            'https://www.googleadservices.com',

            // Google Tag Manager
            'https://www.googletagmanager.com',

            // Google Tag Manager's Preview Mode
            'https://tagmanager.google.com',

            // Universal Analytics (Google Analytics)
            'https://ssl.google-analytics.com',
            'https://www.google-analytics.com',
          ]),
          styleSrc: lodashUniq([
            "'self'",

            // Google Tag Manager's Preview Mode
            'https://fonts.googleapis.com',
            'https://tagmanager.google.com',
          ]),

          /* Document directives */
          baseUri: ["'none'"],
          /*
           * To disallow all plugins, the object-src directive should be set to 'none' which will disallow plugins.
           * The plugin-types directive is only used if you are allowing plugins with object-src at all.
           * pluginTypes: [],
           */
          sandbox: ['allow-popups', 'allow-same-origin', 'allow-scripts'],

          /* Navigation directives */
          formAction: ["'none'"],
          frameAncestors: ["'none'"],

          /* Reporting directives */
          reportUri: cspReportURI,

          /* Other directives */
          /*
           * The upgrade-insecure-requests directive is evaluated before block-all-mixed-content and if it is set,
           * the latter is effectively a no-op. It is recommended to set either directive, but not both,
           * unless you want to force HTTPS on older browsers that do not force it after a redirect to HTTP.
           * blockAllMixedContent: [],
           */
          upgradeInsecureRequests: [],
        },
      },

      // Set header 'Referrer-Policy: no-referrer'
      referrerPolicy: {
        policy: 'no-referrer',
      },

      // Set header 'Strict-Transport-Security: max-age=31536000; includeSubDomains; preload'
      hsts: {
        // Must be at least 1 year to be approved
        maxAge: 31536000,

        // Must be enabled to be approved
        includeSubDomains: true,
        preload: true,
      },

      // Set header 'X-Content-Type-Options: nosniff'
      noSniff: undefined,

      // Set header 'X-DNS-Prefetch-Control: off'
      dnsPrefetchControl: {
        allow: false,
      },

      // Set header 'X-Download-Options: noopen'
      ieNoOpen: undefined,

      // Set header 'X-Frame-Options: DENY'
      frameguard: {
        action: 'deny',
      },

      // Set header 'X-Permitted-Cross-Domain-Policies: none'
      permittedCrossDomainPolicies: {
        permittedPolicies: 'none',
      },

      // Remove header 'X-Powered-By'
      hidePoweredBy: undefined,

      // Set header 'X-XSS-Protection: 0'
      xssFilter: undefined,
    })(req, res, next);
  };
};

export default helmetMiddleware;
