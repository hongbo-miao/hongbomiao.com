const {
  DOMAIN,
  LIGHTSTEP_TOEKN,
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
  port: PORT,

  devCSPConnectSrc: [...sharedCSPConnectSrc],
  prodCSPConnectSrc: [...sharedCSPConnectSrc],

  devCORSAllowList: [
    ...sharedCORSAllowList,
    'https://localhost:5000',
    'https://localhost:5001',
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
    token: LIGHTSTEP_TOEKN,
    traceURL: 'https://ingest.lightstep.com:443/api/v2/otel/trace',
  },

  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },
};

export default Config;
