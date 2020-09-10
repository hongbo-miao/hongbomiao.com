import dotenvFlow from 'dotenv-flow';

dotenvFlow.config();

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
  REDIS_PORT,
  REDIS_PASSWORD,
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
if (REDIS_PORT == null || REDIS_PORT === '') {
  throw new Error('Failed to read REDIS_PORT.');
}
if (REDIS_PASSWORD == null) {
  throw new Error('Failed to read REDIS_PASSWORD.');
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

const Config = {
  nodeEnv: NODE_ENV,
  domain: DOMAIN,
  port: Number(PORT),
  externalPort: 443,

  devCSPConnectSrc: [...sharedCSPConnectSrc, 'https://localhost:443'],
  prodCSPConnectSrc: [...sharedCSPConnectSrc],

  devCORSAllowList: [
    ...sharedCORSAllowList,
    'https://localhost:443',
    'https://localhost:5000',
    'https://localhost:8080',
  ],
  prodCORSAllowList: [...sharedCORSAllowList, 'https://hongbomiao.com', 'https://www.hongbomiao.com'],

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

export default Config;
