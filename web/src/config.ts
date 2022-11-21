import * as Sentry from '@sentry/react';
import { BrowserTracing } from '@sentry/tracing';

const { NODE_ENV, REACT_APP_SERVER_WS_PROTOCOL } = process.env;

if (NODE_ENV !== 'development' && NODE_ENV !== 'production' && NODE_ENV !== 'test') {
  throw new Error('Failed to read NODE_ENV.');
}
if (REACT_APP_SERVER_WS_PROTOCOL == null || REACT_APP_SERVER_WS_PROTOCOL === '') {
  throw new Error('Failed to read REACT_APP_SERVER_WS_PROTOCOL.');
}

type Config = {
  nodeEnv: 'development' | 'production' | 'test';
  githubURL: string;
  graphqlServerGraphQLURL: string;
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
  graphqlServerGraphQLURL: NODE_ENV === 'development' ? 'http://localhost:31800/graphql' : '/graphql-server/graphql',
  webSocketGraphQLURL: `${REACT_APP_SERVER_WS_PROTOCOL}://${window.location.host}/graphql`,
  googleTagManagerOptions: {
    containerId: 'GTM-MKMQ55P',
  },
  lightstep: {
    token: 'W2sFPG0uCgnCAjr/d0NfngMArOSUEb1SN/5UlOLnZxQ3/4hWndgg/J3jZX74b/c0AF4+o+h0lRGY2vHHFWJBuMh4CKMyILo3pMznB4xd',
    traceURL: 'https://ingest.lightstep.com/api/v2/otel/trace',
  },
  sentryOptions: {
    dsn: 'https://a0ff55d9ee00403ca144425a33c318eb@o379185.ingest.sentry.io/4504195581018112',
    environment: NODE_ENV,
    integrations: [new BrowserTracing()],
    tracesSampleRate: 1.0,
  },
};

export default config;
