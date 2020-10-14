import * as Sentry from '@sentry/react';

const { REACT_APP_SERVER_DOMAIN, REACT_APP_LIGHTSTEP_TOKEN, REACT_APP_SERVER_PORT, NODE_ENV } = process.env;

if (NODE_ENV !== 'development' && NODE_ENV !== 'production' && NODE_ENV !== 'test') {
  throw new Error('Failed to read NODE_ENV.');
}
if (REACT_APP_SERVER_DOMAIN == null || REACT_APP_SERVER_DOMAIN === '') {
  throw new Error('Failed to read REACT_APP_SERVER_DOMAIN.');
}
if (REACT_APP_SERVER_PORT == null || REACT_APP_SERVER_PORT === '') {
  throw new Error('Failed to read REACT_APP_SERVER_PORT.');
}

type Config = {
  nodeEnv: 'development' | 'production' | 'test';
  githubURL: string;
  graphQLURL: string;
  webSocketGraphQLURL: string;
  googleTagManagerOptions: {
    containerId: string;
  };
  lightstep: {
    token: string | undefined;
    traceURL: string;
  };
  sentryOptions: Sentry.BrowserOptions;
};

const config: Config = {
  nodeEnv: NODE_ENV,
  githubURL: 'https://github.com/Hongbo-Miao/hongbomiao.com',
  graphQLURL: `https://${REACT_APP_SERVER_DOMAIN}:${REACT_APP_SERVER_PORT}/graphql`,
  webSocketGraphQLURL: `wss://${REACT_APP_SERVER_DOMAIN}:${REACT_APP_SERVER_PORT}/graphql`,
  googleTagManagerOptions: {
    containerId: 'GTM-MKMQ55P',
  },
  lightstep: {
    token: REACT_APP_LIGHTSTEP_TOKEN,
    traceURL: 'https://ingest.lightstep.com:443/api/v2/otel/trace',
  },
  sentryOptions: {
    dsn: 'https://7a5703c6beeb467e9cdb314cec25a237@o379185.ingest.sentry.io/5384453',
    environment: NODE_ENV,
  },
};

export default config;
