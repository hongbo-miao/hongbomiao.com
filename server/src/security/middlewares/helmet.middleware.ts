import fs from 'fs';
import path from 'path';
import { RequestHandler } from 'express';
import helmet from 'helmet';
import Config from '../../Config';
import createScriptSrcHashes from '../utils/createScriptSrcHashes';

const SCRIPT_SRC_HASHES = createScriptSrcHashes(
  fs.readFileSync(path.join(__dirname, '../../../dist/index.html'), 'utf-8')
);

const helmetMiddleware = (scriptSrcHashes: string[] = SCRIPT_SRC_HASHES): RequestHandler => {
  return helmet({
    // Set 'Content-Security-Policy'
    contentSecurityPolicy: {
      directives: {
        /* Fetch directives */
        connectSrc: [
          "'self'",
          'https://hongbomiao.herokuapp.com',
          'https://rs.fullstory.com',
          'https://stats.g.doubleclick.net',
          'https://www.google-analytics.com',
        ],
        defaultSrc: ["'none'"],
        fontSrc: ["'self'", 'data:', 'https://fonts.gstatic.com'],
        frameSrc: ["'none'"],
        imgSrc: [
          "'self'",
          'data:',
          'https://stats.g.doubleclick.net',
          'https://www.google-analytics.com',
          'https://www.google.com',
        ],
        manifestSrc: ["'self'"],
        mediaSrc: ["'none'"],
        objectSrc: ["'none'"],
        scriptSrc: [
          "'self'",
          'https://edge.fullstory.com',
          'https://fullstory.com',
          'https://storage.googleapis.com',
          'https://www.google-analytics.com',
          'https://www.googletagmanager.com',
          "'sha256-KYbqXqOYv6xOvzbfRwXslg+GK2ebpVi0G6EzAvF6Yc8='", // Google Tag Manager
          ...scriptSrcHashes,
        ],
        styleSrc: ["'self'", 'https://fonts.googleapis.com'],

        /* Document directives */
        baseUri: ["'none'"],
        // pluginTypes: [], // To disallow all plugins, the object-src directive should be set to 'none' which will disallow plugins. The plugin-types directive is only used if you are allowing plugins with object-src at all.
        sandbox: ['allow-same-origin', 'allow-scripts'],

        /* Navigation directives */
        formAction: ["'none'"],
        frameAncestors: ["'none'"],

        /* Reporting directives */
        reportUri: `https://${Config.domain}:${Config.port}/api/violation/report-csp-violation`,

        /* Other directives */
        // blockAllMixedContent: [], // The upgrade-insecure-requests directive is evaluated before block-all-mixed-content and if it is set, the latter is effectively a no-op. It is recommended to set either directive, but not both, unless you want to force HTTPS on older browsers that do not force it after a redirect to HTTP.
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
  });
};

export default helmetMiddleware;
