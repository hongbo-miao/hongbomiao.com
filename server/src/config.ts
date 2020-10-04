import * as Sentry from '@sentry/node';
import dotenvFlow from 'dotenv-flow';
import Redis from 'ioredis';
import knex from 'knex';
import { argv } from 'yargs';
import PostgresInputUser from './database/postgres/types/PostgresInputUser.type';

dotenvFlow.config();

const { hideHTTPLog, prettifyLog } = argv;

if (hideHTTPLog != null && typeof hideHTTPLog !== 'boolean') {
  throw new Error('Failed to read hideHTTPLog.');
}
if (prettifyLog != null && typeof prettifyLog !== 'boolean') {
  throw new Error('Failed to read prettifyLog.');
}

const {
  DOMAIN,
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
if (NODE_ENV !== 'development' && NODE_ENV !== 'production' && NODE_ENV !== 'test') {
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

const sharedCSPConnectSrc = [
  'https://ingest.lightstep.com', // Lightstep
];
const sharedCORSAllowList = [
  'electron://altair', // Altair GraphQL Client
  'null', // Safari reports CSP violation
];

type Config = {
  shouldHideHTTPLog: boolean;
  shouldPrettifyLog: boolean;

  nodeEnv: 'development' | 'production' | 'test';
  domain: string;
  port: number;
  devCORSAllowList: ReadonlyArray<string>;
  prodCORSAllowList: ReadonlyArray<string>;
  devCSPConnectSrc: ReadonlyArray<string>;
  prodCSPConnectSrc: ReadonlyArray<string>;
  reportURI: {
    cspReportURI: string;
    exceptCTReportURI: string;
    reportToURL: string;
  };
  redisOptions: Redis.RedisOptions;
  postgresConnection: knex.StaticConnectionConfig;
  seedUser: PostgresInputUser;
  lightstep: {
    token: string | undefined;
    traceURL: string;
  };
  sentryOptions: Sentry.NodeOptions;
};

const config: Config = {
  shouldHideHTTPLog: hideHTTPLog === true,
  shouldPrettifyLog: prettifyLog === true,

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
  devCSPConnectSrc: [
    ...sharedCSPConnectSrc,

    // For Safari, "'self'" is not enough for WebSocket.
    'wss://localhost:443',
    'wss://localhost:5000',
  ],
  prodCSPConnectSrc: [
    ...sharedCSPConnectSrc,

    // For Safari, "'self'" is not enough for WebSocket.
    'wss://www.hongbomiao.com',
  ],
  reportURI: {
    cspReportURI: 'https://hongbomiao.report-uri.com/r/d/csp/enforce',
    exceptCTReportURI: 'https://hongbomiao.report-uri.com/r/d/ct/enforce',
    reportToURL: 'https://hongbomiao.report-uri.com/a/d/g',
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
};

export default config;
