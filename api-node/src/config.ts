import * as Sentry from '@sentry/node';
import dotenvFlow from 'dotenv-flow';
import { RedisOptions } from 'ioredis';
import { Knex } from 'knex';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import PostgresInputUser from './dataSources/postgres/types/PostgresInputUser';

dotenvFlow.config();

const {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  argv: { hideHTTPLog, prettifyLog },
} = yargs(hideBin(process.argv));

if (hideHTTPLog != null && typeof hideHTTPLog !== 'boolean') {
  throw new Error('Failed to read hideHTTPLog.');
}
if (prettifyLog != null && typeof prettifyLog !== 'boolean') {
  throw new Error('Failed to read prettifyLog.');
}

const {
  HOST,
  HTTP_PROTOCOL,
  JWT_SECRET,
  NODE_ENV,
  PORT,
  POSTGRES_DB,
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
  WS_PROTOCOL,
} = process.env;

if (HOST == null || HOST === '') {
  throw new Error('Failed to read HOST.');
}
if (HTTP_PROTOCOL == null || HTTP_PROTOCOL === '') {
  throw new Error('Failed to read HTTP_PROTOCOL.');
}
if (JWT_SECRET == null || JWT_SECRET === '') {
  throw new Error('Failed to read JWT_SECRET.');
}
if (NODE_ENV !== 'development' && NODE_ENV !== 'production' && NODE_ENV !== 'test') {
  throw new Error('Failed to read NODE_ENV.');
}
if (PORT == null || PORT === '') {
  throw new Error('Failed to read PORT.');
}
if (POSTGRES_DB == null || POSTGRES_DB === '') {
  throw new Error('Failed to read POSTGRES_DB.');
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
if (WS_PROTOCOL == null || WS_PROTOCOL === '') {
  throw new Error('Failed to read WS_PROTOCOL.');
}

const sharedCSPConnectSrc = [
  `${WS_PROTOCOL}://${HOST}:${PORT}`, // For Safari, "'self'" is not enough for WebSocket.
  'https://ingest.lightstep.com', // Lightstep
];
const sharedCORSAllowOrigins = [
  `${HTTP_PROTOCOL}://${HOST}:${PORT}`,
  'electron://altair', // Altair GraphQL Client
  'null', // Safari reports CSP violation
];

type Config = {
  shouldHideHTTPLog: boolean;
  shouldPrettifyLog: boolean;
  nodeEnv: 'development' | 'production' | 'test';
  httpProtocol: string;
  port: number;
  devCORSAllowOrigins: ReadonlyArray<string>;
  prodCORSAllowOrigins: ReadonlyArray<string>;
  devCSPConnectSrc: ReadonlyArray<string>;
  prodCSPConnectSrc: ReadonlyArray<string>;
  reportURI: {
    cspReportURI: string;
    exceptCTReportURI: string;
    reportToURL: string;
  };
  jwtSecret: string;
  redisOptions: RedisOptions;
  postgresConnection: Knex.StaticConnectionConfig;
  seedUser: PostgresInputUser;
  lightstep: {
    token: string | undefined;
    traceURL: string;
  };
  sentryOptions: Sentry.NodeOptions;
  swapiURL: string;
};

const config: Config = {
  shouldHideHTTPLog: hideHTTPLog === true,
  shouldPrettifyLog: prettifyLog === true,
  nodeEnv: NODE_ENV,
  httpProtocol: HTTP_PROTOCOL,
  port: Number(PORT),
  devCORSAllowOrigins: [
    ...sharedCORSAllowOrigins,
    `${HTTP_PROTOCOL}://${HOST}:80`,
    `${HTTP_PROTOCOL}://${HOST}:3000`,
    'https://www.k8s-hongbomiao.com',
  ],
  prodCORSAllowOrigins: [...sharedCORSAllowOrigins, 'https://www.hongbomiao.com'],
  devCSPConnectSrc: [...sharedCSPConnectSrc, `${WS_PROTOCOL}://${HOST}:80`, 'wss://www.k8s-hongbomiao.com'],
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
  jwtSecret: JWT_SECRET,
  redisOptions: {
    host: REDIS_HOST,
    port: Number(REDIS_PORT),
    password: REDIS_PASSWORD,
    enableOfflineQueue: false,
  },
  postgresConnection: {
    host: POSTGRES_HOST,
    port: Number(POSTGRES_PORT),
    database: POSTGRES_DB,
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
    token: '+vWuQYt3zPOA+eurdLSoF5ksPKxwso69BKMecNlBUj+JkdAts+nwEOOVYHMG0AWIUUPbFX3JM/Z3MkaA9bWU/+16mluwwryRtmIz3Gnx',
    traceURL: 'https://ingest.lightstep.com/api/v2/otel/trace',
  },
  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },
  swapiURL: 'https://swapi.dev',
};

export default config;
