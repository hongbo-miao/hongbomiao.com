import * as Sentry from '@sentry/react';

const {
  NODE_ENV,
  REACT_APP_LIGHTSTEP_TOKEN,
  REACT_APP_SERVER_HOST,
  REACT_APP_SERVER_HTTP_PROTOCOL,
  REACT_APP_SERVER_PORT,
  REACT_APP_SERVER_WS_PROTOCOL,
} = process.env;

if (NODE_ENV !== 'development' && NODE_ENV !== 'production' && NODE_ENV !== 'test') {
  throw new Error('Failed to read NODE_ENV.');
}
if (REACT_APP_SERVER_HOST == null || REACT_APP_SERVER_HOST === '') {
  throw new Error('Failed to read REACT_APP_SERVER_HOST.');
}
if (REACT_APP_SERVER_HTTP_PROTOCOL == null || REACT_APP_SERVER_HTTP_PROTOCOL === '') {
  throw new Error('Failed to read REACT_APP_SERVER_HTTP_PROTOCOL.');
}
if (REACT_APP_SERVER_PORT == null || REACT_APP_SERVER_PORT === '') {
  throw new Error('Failed to read REACT_APP_SERVER_PORT.');
}
if (REACT_APP_SERVER_WS_PROTOCOL == null || REACT_APP_SERVER_WS_PROTOCOL === '') {
  throw new Error('Failed to read REACT_APP_SERVER_WS_PROTOCOL.');
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
  graphQLURL: `${REACT_APP_SERVER_HTTP_PROTOCOL}://${REACT_APP_SERVER_HOST}:${REACT_APP_SERVER_PORT}/graphql`,
  webSocketGraphQLURL: `${REACT_APP_SERVER_WS_PROTOCOL}://${REACT_APP_SERVER_HOST}:${REACT_APP_SERVER_PORT}/graphql`,
  googleTagManagerOptions: {
    containerId: 'GTM-MKMQ55P',
  },
  lightstep: {
    token: REACT_APP_LIGHTSTEP_TOKEN,
    traceURL: 'https://ingest.lightstep.com/api/v2/otel/trace',
  },
  sentryOptions: {
    dsn: 'https://2f46725646834700b4c2675abbc2da6a@o379185.ingest.sentry.io/5375232',
    environment: NODE_ENV,
  },
};

export default config;
