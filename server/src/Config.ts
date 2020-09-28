import dotenvFlow from 'dotenv-flow';
import { argv } from 'yargs';
import NodeEnv from './shared/utils/NodeEnv';

dotenvFlow.config();

const {
  DOMAIN,
  FLUENT_BIT_HOST,
  FLUENT_BIT_PORT,
  LIGHTSTEP_TOKEN,
  NODE_ENV,
  PORT,
  POSTGRES_DATABASE,
  POSTGRES_HOST,
  POSTGRES_PASSWORD,
  POSTGRES_PORT,
  POSTGRES_USER,
  REDIS_HOST,
  REDIS_PASSWORD,
  REDIS_PORT,
  SEED_USER_BIO,
  SEED_USER_EMAIL,
  SEED_USER_FIRST_NAME,
  SEED_USER_LAST_NAME,
  SEED_USER_PASSWORD,
} = process.env;

if (DOMAIN == null || DOMAIN === '') {
  throw new Error('Failed to read DOMAIN.');
}
if (LIGHTSTEP_TOKEN == null || LIGHTSTEP_TOKEN === '') {
  throw new Error('Failed to read LIGHTSTEP_TOKEN.');
}
if (NODE_ENV == null || NODE_ENV === '') {
  throw new Error('Failed to read NODE_ENV.');
}
if (PORT == null || PORT === '') {
  throw new Error('Failed to read PORT.');
}
if (POSTGRES_DATABASE == null || POSTGRES_DATABASE === '') {
  throw new Error('Failed to read POSTGRES_DATABASE.');
}
if (POSTGRES_HOST == null || POSTGRES_HOST === '') {
  throw new Error('Failed to read POSTGRES_HOST.');
}
if (POSTGRES_PASSWORD == null || POSTGRES_PASSWORD === '') {
  throw new Error('Failed to read POSTGRES_PASSWORD.');
}
if (POSTGRES_PORT == null || POSTGRES_PORT === '') {
  throw new Error('Failed to read POSTGRES_PORT.');
}
if (POSTGRES_USER == null || POSTGRES_USER === '') {
  throw new Error('Failed to read POSTGRES_USER.');
}
if (REDIS_HOST == null || REDIS_HOST === '') {
  throw new Error('Failed to read REDIS_HOST.');
}
if (REDIS_PASSWORD == null) {
  throw new Error('Failed to read REDIS_PASSWORD.');
}
if (REDIS_PORT == null || REDIS_PORT === '') {
  throw new Error('Failed to read REDIS_PORT.');
}
if (SEED_USER_BIO == null || SEED_USER_BIO === '') {
  throw new Error('Failed to read SEED_USER_BIO.');
}
if (SEED_USER_EMAIL == null || SEED_USER_EMAIL === '') {
  throw new Error('Failed to read SEED_USER_EMAIL.');
}
if (SEED_USER_FIRST_NAME == null || SEED_USER_FIRST_NAME === '') {
  throw new Error('Failed to read SEED_USER_FIRST_NAME.');
}
if (SEED_USER_LAST_NAME == null || SEED_USER_LAST_NAME === '') {
  throw new Error('Failed to read SEED_USER_LAST_NAME.');
}
if (SEED_USER_PASSWORD == null || SEED_USER_PASSWORD === '') {
  throw new Error('Failed to read SEED_USER_PASSWORD.');
}

if (NODE_ENV === NodeEnv.development) {
  if (FLUENT_BIT_HOST == null || FLUENT_BIT_HOST === '') {
    throw new Error('Failed to read FLUENT_BIT_HOST.');
  }
  if (FLUENT_BIT_PORT == null || FLUENT_BIT_PORT === '') {
    throw new Error('Failed to read FLUENT_BIT_PORT.');
  }
}

if (argv.log != null && typeof argv.log !== 'boolean') {
  throw new Error('Failed to read argv log.');
}

const sharedCSPConnectSrc = [
  'https://ingest.lightstep.com', // Lightstep
];
const sharedCORSAllowList = [
  'electron://altair', // Altair GraphQL Client
  'null', // Safari reports CSP violation
];

const Config = {
  shouldShowLog: !!argv.log,

  nodeEnv: NODE_ENV,
  domain: DOMAIN,
  port: Number(PORT),

  devCORSAllowList: [
    ...sharedCORSAllowList,
    'https://localhost:443',
    'https://localhost:5000',
    'https://localhost:8080',
  ],
  prodCORSAllowList: [...sharedCORSAllowList, 'https://www.hongbomiao.com'],

  devCSPConnectSrc: [...sharedCSPConnectSrc, 'https://localhost:443'],
  prodCSPConnectSrc: [...sharedCSPConnectSrc],

  reportURI: {
    cspReportUri: 'https://hongbomiao.report-uri.com/r/d/csp/enforce',
    exceptCtReportUri: 'https://hongbomiao.report-uri.com/r/d/ct/enforce',
  },

  redisOptions: {
    host: REDIS_HOST,
    port: Number(REDIS_PORT),
    password: REDIS_PASSWORD,
    enableOfflineQueue: false,
  },

  postgresConnection: {
    host: POSTGRES_HOST,
    port: Number(POSTGRES_PORT),
    database: POSTGRES_DATABASE,
    user: POSTGRES_USER,
    password: POSTGRES_PASSWORD,
  },

  seedUser: {
    email: SEED_USER_EMAIL,
    password: SEED_USER_PASSWORD,
    firstName: SEED_USER_FIRST_NAME,
    lastName: SEED_USER_LAST_NAME,
    bio: SEED_USER_BIO,
  },

  lightstep: {
    token: LIGHTSTEP_TOKEN,
    traceURL: 'https://ingest.lightstep.com:443/api/v2/otel/trace',
  },

  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },

  fluentBitConfig: {
    host: FLUENT_BIT_HOST,
    port: Number(FLUENT_BIT_PORT),
    timeout: 3.0,
    requireAckResponse: true, // Add this option to wait response from Fluent Bit certainly
  },
};

export default Config;
