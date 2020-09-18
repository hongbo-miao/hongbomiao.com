import { NextFunction, Request, RequestHandler, Response } from 'express';
import helmet from 'helmet';
import Config from '../../Config';
import isProduction from '../../shared/utils/isProduction';
import createCSPNonce from '../utils/createCSPNonce';

const CSP_CONNECT_SRC = isProduction() ? Config.prodCSPConnectSrc : Config.devCSPConnectSrc;
const EXTERNAL_PORT = isProduction() ? Config.externalPort : Config.port;

const helmetMiddleware = (
  cspConnectSrc: string[] = CSP_CONNECT_SRC,
  externalPort: number = EXTERNAL_PORT
): RequestHandler => {
  return (req: Request, res: Response, next: NextFunction) => {
    const cspNonce = createCSPNonce();
    res.locals.cspNonce = cspNonce;

    return helmet({
      // Set 'Content-Security-Policy'
      contentSecurityPolicy: {
        directives: {
          /* Fetch directives */
          connectSrc: [
            "'self'",

            // FullStory
            'https://rs.fullstory.com',

            // LogRocket
            'https://r.lr-ingest.io',

            // Sentry
            'https://o379185.ingest.sentry.io',

            // Universal Analytics (Google Analytics)
            'https://www.google-analytics.com',
            'https://stats.g.doubleclick.net',

            ...cspConnectSrc,
          ],
          defaultSrc: ["'none'"],
          fontSrc: [
            "'self'",
            'data:',

            // Google Tag Manager's Preview Mode
            'https://fonts.gstatic.com',
          ],
          frameSrc: ["'none'"],
          imgSrc: [
            "'self'",
            'data:',

            // Google Tag Manager's Preview Mode
            'https://www.googletagmanager.com',
            'https://ssl.gstatic.com',

            // Universal Analytics (Google Analytics)
            'https://www.google-analytics.com',
            'https://stats.g.doubleclick.net',
          ],
          manifestSrc: ["'self'"],
          mediaSrc: ["'none'"],
          objectSrc: ["'none'"],
          scriptSrc: [
            /* Content Security Policy Level 3 */
            "'strict-dynamic'",
            `'nonce-${cspNonce}'`,

            /* Content Security Policy Level 2 (backward compatible) */
            "'self'",

            // Cloudflare Browser Insights
            'https://static.cloudflareinsights.com',

            // FullStory
            'https://edge.fullstory.com',
            'https://fullstory.com',

            // LogRocket
            'https://cdn.lr-ingest.io',

            // Workbox
            'https://storage.googleapis.com',

            // Google Tag Manager
            'https://www.googletagmanager.com',

            // Google Tag Manager's Preview Mode
            'https://tagmanager.google.com',

            // Universal Analytics (Google Analytics)
            'https://www.google-analytics.com',
            'https://ssl.google-analytics.com',
          ],
          styleSrc: [
            "'self'",

            // Google Tag Manager's Preview Mode
            'https://tagmanager.google.com',
            'https://fonts.googleapis.com',
          ],
          workerSrc: ["'self'"],

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
          reportUri: `https://${Config.domain}:${externalPort}/api/violation/report-csp-violation`,

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
};

export default helmetMiddleware;
