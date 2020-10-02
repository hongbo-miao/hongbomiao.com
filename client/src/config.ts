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
  lightstep: {
    token: REACT_APP_LIGHTSTEP_TOKEN,
    traceURL: 'https://ingest.lightstep.com:443/api/v2/otel/trace',
  },
  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },
};

export default config;
