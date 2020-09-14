const { REACT_APP_SERVER_DOMAIN, REACT_APP_LIGHTSTEP_TOKEN, REACT_APP_SERVER_PORT, NODE_ENV } = process.env;

if (NODE_ENV == null) {
  throw new Error('Failed to read NODE_ENV.');
}
if (REACT_APP_SERVER_DOMAIN == null || REACT_APP_SERVER_DOMAIN === '') {
  throw new Error('Failed to read REACT_APP_SERVER_DOMAIN.');
}
if (REACT_APP_SERVER_PORT == null || REACT_APP_SERVER_PORT === '') {
  throw new Error('Failed to read REACT_APP_SERVER_PORT.');
}

const Config = {
  nodeEnv: NODE_ENV,

  githubURL: 'https://github.com/hongbo-miao/hongbomiao.com',
  graphQLURL: `https://${REACT_APP_SERVER_DOMAIN}:${REACT_APP_SERVER_PORT}/graphql`,

  lightstep: {
    token: REACT_APP_LIGHTSTEP_TOKEN,
    traceURL: 'https://ingest.lightstep.com:443/api/v2/otel/trace',
  },

  logRocketAppID: 'hongbo-miao/hongbomiao',

  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },
};

export default Config;
